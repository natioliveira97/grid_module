"""
Created by: Anshuman Dewangan
Date: 2021

Description: PyTorch Lightning LightningModule that defines optimizers, training step and metrics.
"""



# Torch imports
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchmetrics
import csv

# Other imports
import numpy as np
import collections

# File imports
from model_components import *
from grid_dataloader import *
  
class MainModel(nn.Module):
    """
    Description: Simple model with ResNet backbone and a few linear layers
    Args (see arg descriptions in main.py):
        - kwargs: any other args used in the models
    """
    def __init__(self, 
                 tile_loss_type='bce',
                 bce_pos_weight=36,
                 focal_alpha=0.25, 
                 focal_gamma=2,
                 image_pos_weight=1,
                 confidence_threshold=0, 
                 model_type_list=[],      
                 **kwargs):
        
        print("Initializing MainModel...")

        super().__init__()
        
        ### Initialize Model ###
        self.model_list = torch.nn.ModuleList()

        # Initializes each model using the class name and kwargs and adds it to model_list
        for model_type in model_type_list:
            self.model_list.append(globals()[model_type](**kwargs))
                
        self.image_pos_weight = image_pos_weight
        self.confidence_threshold = confidence_threshold

        ### Initialize Loss ###
        self.tile_loss = TileLoss(tile_loss_type=tile_loss_type,
                                  bce_pos_weight=bce_pos_weight,
                                  focal_alpha=focal_alpha, 
                                  focal_gamma=focal_gamma)
                
        print("Initializing MainModel Complete.")
        
    def forward(self, x):
        """Description: Maps raw inputs to outputs"""
        outputs = None
        
        for model in self.model_list:
            outputs, x = model(x, outputs)
            
        return outputs
        
    def forward_pass(self, x, tile_labels, ground_truth_labels, split, num_epoch, device, outputs=None):
        """
        Description: compute forward pass of all model_list models
        Args:
            - x (tensor): raw image input
            - tile_labels (tensor): labels for tiles for tile_loss
            - bbox_labels: (list): labels for bboxes
            - ground_truth_labels (tensor): labels for images for image_loss
            - omit_masks (tensor): determines if tile predictions should be masked
            - split (str): if this is the train/val/test split for determining correct loss calculation
            - num_epoch (int): current epoch number (for pretrain_epochs)
            - device: current device being used to create new torch objects without getting errors
        Outputs:
            - losses (list of tensor): all tile-level or image-level losses (for logging purposes)
            - image_loss (list of tensor): all image-level losses (for logging purposes)
            - total_loss (tensor): overall loss (sum of all losses)
            - tile_probs (tensor): probabilities predicted for each tile
            - tile_preds (tensor): final predictions for each tile
            - image_preds (tensor): final predictions for each image
        """
        # print(tile_labels, ground_truth_labels, split, num_epoch, device, outputs,self.model_list)
       
        tile_outputs = None
        image_outputs = None
        image_probs=np.zeros(x.shape[0])
        
        losses = []
        total_loss = 0
        image_loss = None
        
        tile_probs = None
        tile_preds = None
        image_preds = None
        
        # Compute forward pass and loss for each model in model_list
        for i, model in enumerate(self.model_list):
            # Skip iteration if pretraining model
            if i != 0:# and self.pretrain_epochs[:i].sum() > num_epoch:
                break
            
            # Compute forward pass
            if outputs is None or i > 0:
                outputs, x = model(x, tile_outputs=outputs)
                        
            # FPN ONLY: If outputs is a dictionary of FPN layers...
            if type(outputs) is collections.OrderedDict:
                returns = {}
                for key in outputs:
                    # Run through forward pass for each layer of FPN
                    # Outputs: (losses, image_loss, total_loss, tile_probs, tile_preds, image_preds)
                    returns[key] = self.forward_pass(x[key], tile_labels, ground_truth_labels, split, num_epoch, device, outputs[key])
                    
                    # Compute correct statistics
                    losses = returns[key][0] if len(losses)==0 else losses + returns[key][0]
                    image_loss = returns[key][1] if image_loss is None else image_loss + returns[key][1]
                    total_loss += returns[key][2] / len(outputs)
                    tile_probs = returns[key][3] if tile_probs is None else tile_probs + returns[key][3]
                    tile_preds = returns[key][4] if tile_preds is None else torch.logical_or(tile_preds, returns[key][4])
                    image_preds = returns[key][5] if image_preds is None else torch.logical_or(image_preds, returns[key][5])
                    
                if tile_probs is not None:
                    tile_probs = tile_probs / len(outputs)
                    
                break
            
            # FLOW ONLY: If outputs is a tuple for both image and flow...
            elif type(outputs) is tuple:
                # Loop through outputs for image and flow
                for output in outputs:
                    tile_outputs = output
                    # loss = self.tile_loss(tile_outputs[omit_masks,:,-1], tile_labels[omit_masks], num_epoch=num_epoch) 

                    # Only add loss if intermediate_supervision
                    if self.intermediate_supervision and not self.image_loss_only:
                        total_loss += loss
                        losses.append(loss)
                    else:
                        total_loss = loss
            
            # OBJECT DETECTION ONLY: If x is a dictionary of object detection losses...
            elif type(x) is dict:
                # If training...
                if len(x) > 0: 
                    # Use losses from model
                    for loss in x:
                        total_loss += x[loss]
                
                # Else for val/test loss...
                else:
                    # Determine if there were any scores above confidence = 0
                    image_preds = torch.as_tensor([(output['scores'] > self.confidence_threshold).sum() > 0 for output in outputs]).to(device)
                    
                    # Use number of errors as loss
                    total_loss = torch.abs(image_preds.float() - ground_truth_labels.float()).sum()
                    
                return losses, image_loss, total_loss, tile_probs, tile_preds, image_preds
            
            # IMAGE ONLY: Else if model predicts images only...
            elif x is None:
                # Calculate image loss
                image_outputs = outputs
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                # Always add image loss, even if no intermediate_supervision
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
                                        
            # TILE ONLY: Else if model predicts tiles only...
            elif len(x.shape) > 2:
                # Calculate tile loss
                tile_outputs = outputs
                # print(tile_outputs.shape)
                # print(tile_labels.shape)
                loss = self.tile_loss(tile_outputs[:,:,-1].float(), tile_labels.float(), num_epoch=num_epoch) 
                # print("tyle_only")
                
                # # Only add loss if intermediate_supervision
                # if self.intermediate_supervision and not self.image_loss_only:
                #     total_loss += loss
                #     losses.append(loss)
                # else:
                total_loss = loss
                
            # TILE & IMAGES: Else if model predicts tiles and images...
            else:
                # Calculate tile loss
                tile_outputs = outputs
                if not self.image_loss_only:
                    loss = self.tile_loss(tile_outputs, tile_labels, num_epoch=num_epoch) 
                    # print("tile_image")
                    total_loss += loss
                    losses.append(loss)
                
                # Calculate image loss
                image_outputs = x
                image_loss = F.binary_cross_entropy_with_logits(image_outputs[:,-1], ground_truth_labels.float(), reduction='none', pos_weight=torch.as_tensor(self.image_pos_weight))
                
                loss = image_loss.mean()
                total_loss += loss
                losses.append(loss)
            
        # Compute predictions for tiles
        if tile_outputs is not None:
            tile_probs = torch.sigmoid(tile_outputs[:,:,-1])
            tile_preds = (tile_probs > 0.5).int()
        
        # If created image_outputs, predict directly
        if image_outputs is not None:
            image_probs = torch.sigmoid(image_outputs[:,-1])
            image_preds = (image_probs > 0.5).int()
        # Else, use tile_preds to determine image_preds
        elif tile_outputs is not None:
            image_preds = (tile_preds.sum(dim=1) > 0).int()
        
        # If error_as_eval_loss, replace total_loss with number of errors
        if  split != 'train/':
            total_loss = torch.abs(image_preds.float() - ground_truth_labels.float()).sum()

        # print(losses, image_loss, total_loss, tile_probs, tile_preds, image_preds)
        return losses, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs


class LightningModule(pl.LightningModule):

    #####################
    ## Initialization
    #####################

    def __init__(self,
                 model,
                 batch_size=2,
                 optimizer_type='SGD',
                 optimizer_weight_decay=0.0001,
                 learning_rate=0.0001,
                 lr_schedule=True,
                 series_length=1,
                 parsed_args=None):
        """
        Args (see arg descriptions in main.py):
            - parsed_args (dict): full dict of parsed args to log as hyperparameters

        Other Attributes:
            - self.metrics (dict): contains many properties related to logging metrics, including:
                - torchmetrics (torchmetrics module): keeps track of metrics per step and per epoch
                - split (list of str): name of splits e.g. ['train/', 'val/', 'test/']
                - category (list of str): metric subcategories
                    - 'tile_': metrics on per-tile basis
                    - 'image-gt': labels based on if image name has '+' in it)
                - name (list of str): name of metric e.g. ['accuracy', 'precision', ...]
        """
        print("Initializing LightningModule...")
        print(batch_size)
        super().__init__()
        
        # Initialize model
        self.model = model
        self.series_length = series_length
        self.batch_size = batch_size
        print(batch_size)
        # Initialize optimizer params
        print(optimizer_type)
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        
        # Save hyperparameters
        self.save_hyperparameters(parsed_args)
        self.save_hyperparameters('learning_rate')
        
        # Initialize evaluation metrics
        self.metrics = {}
        self.metrics['torchmetric'] = {}
        self.metrics['split']       = ['train/', 'val/', 'test/']
        self.metrics['category']    = ['tile_', 'image-gt_']
        self.metrics['name']        = ['accuracy', 'precision', 'recall', 'f1']
        
        for split in self.metrics['split']:
            for i, category in enumerate(self.metrics['category']):
                # Use mdmc_average='global' for tile_preds only
                mdmc_average='global' if i == 0 else None
                
                self.metrics['torchmetric'][split+category+self.metrics['name'][0]] = torchmetrics.Accuracy(task='binary',mdmc_average=mdmc_average)
                self.metrics['torchmetric'][split+category+self.metrics['name'][1]] = torchmetrics.Precision(task='binary',multiclass=False, mdmc_average=mdmc_average)
                self.metrics['torchmetric'][split+category+self.metrics['name'][2]] = torchmetrics.Recall(task='binary',multiclass=False, mdmc_average=mdmc_average)
                self.metrics['torchmetric'][split+category+self.metrics['name'][3]] = torchmetrics.F1Score(task='binary',multiclass=False, mdmc_average=mdmc_average)
            
        print("Initializing LightningModule Complete.")

    def configure_optimizers(self):
        print("optimizer", self.optimizer_type)
        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr=self.learning_rate, 
                                        momentum=0.9, 
                                        weight_decay=self.optimizer_weight_decay)
            print('Optimizer: SGD')
        elif self.optimizer_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                          lr=self.learning_rate, 
                                          weight_decay=self.optimizer_weight_decay)
            print('Optimizer: AdamW')
        else:
            raise ValueError('Optimizer not recognized.')
            
        print('Learning Rate: ', self.learning_rate)
        
        if self.lr_schedule:
            # Includes learning rate scheduler
            scheduler = ReduceLROnPlateau(optimizer, 
                                          min_lr=0, 
                                          factor=0.5,
                                          patience=0,
                                          threshold=0.01,
                                          cooldown=1,
                                          verbose=True)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val/loss"}
        else:
            return optimizer

    def forward(self, x):
        return self.model(x)
        
    #####################
    ## Step Functions
    #####################

    def step(self, batch, split):
        """Description: Takes a batch, calculates forward pass, losses, and predictions, and logs metrics"""
        image_names, x, tile_labels, ground_truth_labels = batch

        # Compute outputs, loss, and predictions
        losses, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs = self.model.forward_pass(x, tile_labels, ground_truth_labels, split, self.current_epoch, self.device)

        # Log intermediate losses (on_step only if split='train')
        for i, loss in enumerate(losses):
            self.log(split+'loss_'+str(i), loss, on_step=(split==self.metrics['split'][0]),on_epoch=True,batch_size=self.batch_size)
        
        # Log overall loss
        self.log(split+'loss', total_loss, on_step=(split==self.metrics['split'][0]),on_epoch=True,batch_size=self.batch_size)

        # Calculate & log evaluation metrics
        for category, args in zip(self.metrics['category'], 
                                  ((tile_preds, tile_labels.int()), 
                                   (image_preds, ground_truth_labels))
                                 ):
            for name in self.metrics['name']:
                # Only log if predictions exist
                if args[0] is not None:
                    pass
                    # Have to move the metric to self.device 
                    self.metrics['torchmetric'][split+category+name].to(self.device)(args[0], args[1])
                    self.log(split+category+name, 
                             self.metrics['torchmetric'][split+category+name], 
                             on_step=False, 
                             on_epoch=True, 
                             metric_attribute=self.metrics['torchmetric'][split+category+name],
                             batch_size=self.batch_size)
        
        return image_names, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs, tile_labels

    def training_step(self, batch, batch_idx):
        image_names, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs, tile_labels = self.step(batch, self.metrics['split'][0])
        return total_loss

    def validation_step(self, batch, batch_idx):
        image_names, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs, tile_labels = self.step(batch, self.metrics['split'][1])

    def test_step(self, batch, batch_idx):
        image_names, image_loss, total_loss, tile_probs, tile_preds, image_preds, image_probs, tile_labels = self.step(batch, self.metrics['split'][2])
        return image_names, image_loss, tile_probs, tile_preds, image_preds, image_probs, tile_labels

    
    #########################
    ## Test Metric Logging
    #########################

    def test_epoch_end(self, test_step_outputs):
        """
        Description: saves predictions to .txt files and computes additional evaluation metrics for test set (e.g. time-to-detection)
        Args:
            - test_step_outputs (list of {image_names, tile_probs, tile_preds, image_preds}): what's returned from test_step
        """
        
        print("Computing Test Evaluation Metrics...")
        positive_preds_dict = {} # Holds positive predictions for each fire
        negative_preds_dict = {} # Holds negative predictions for each fire
        positive_times_dict = {} # Holds timestamps of positives for each fire
        
        ### Save predictions as .txt files ###
        if self.logger is not None:
            image_preds_csv = open(self.logger.log_dir+'/image_preds.csv', 'w')
            image_preds_csv_writer = csv.writer(image_preds_csv)

        # Loop through batch
        for image_names, image_losses, tile_probs, tile_preds, image_preds, image_probs, tile_labels in test_step_outputs:
            # Account for missing predictions (depending on which model is used)
            if tile_probs is None: tile_probs = [None] * len(image_names)
            if tile_preds is None: tile_preds = [None] * len(image_names)
            if image_losses is None: image_losses = [None] * len(image_names)
            
            # Loop through entry in batch
            for image_name, image_loss, tile_prob, tile_pred, image_pred, image_prob, tile_label in zip(image_names, image_losses, tile_probs, tile_preds, image_preds, image_probs, tile_labels):
                # fire_name = util_fns.get_fire_name(image_name)
                fire_name=image_name.split(".mp4")[0]
                image_pred = image_pred.item()
                image_prob = image_prob.item()
                image_loss = image_loss.item() if image_loss else None

                if self.logger is not None:
                    # Save image predictions and image_loss
                    image_preds_csv_writer.writerow([image_name, image_pred, image_prob, image_loss])
                    
                    # Save tile probabilities - useful when visualizing model performance
                    if tile_prob is not None:
                        tile_probs_path = self.logger.log_dir+'/tile_probs/'+fire_name
                        Path(tile_probs_path).mkdir(parents=True, exist_ok=True)
                        np.save(self.logger.log_dir+'/tile_probs/'+\
                                image_name+\
                                '.npy', tile_prob.cpu().numpy())

                    # Save tile predictions - useful when visualizing model performance
                    if tile_pred is not None:
                        tile_preds_path = self.logger.log_dir+'/tile_preds/'+fire_name
                        Path(tile_preds_path).mkdir(parents=True, exist_ok=True)
                        np.save(self.logger.log_dir+'/tile_preds/'+\
                                image_name+\
                                '.npy', tile_pred.cpu().numpy())

                # Add prediction to positive_preds_dict or negative_preds_dict 
                # ASSUMPTION: images are in order and test data has not been shuffled
                if fire_name not in positive_preds_dict:
                    positive_preds_dict[fire_name] = []
                    negative_preds_dict[fire_name] = []
                    positive_times_dict[fire_name] = []

                if util_fns.get_ground_truth_label(image_name) == 0:
                    negative_preds_dict[fire_name].append(image_pred)
                else:
                    positive_preds_dict[fire_name].append(image_pred)
                    positive_times_dict[fire_name].append(util_fns.image_name_to_time_int(image_name))
        
        if self.logger is None: 
            print("No logger. Skipping calculating metrics.")
            return
            
        image_preds_csv.close()

        ### Compute & log metrics ###
        self.log(self.metrics['split'][2]+'negative_accuracy',
                 util_fns.calculate_negative_accuracy(negative_preds_dict),batch_size=self.batch_size)
        self.log(self.metrics['split'][2]+'negative_accuracy_by_fire',
                 util_fns.calculate_negative_accuracy_by_fire(negative_preds_dict),batch_size=self.batch_size)

        self.log(self.metrics['split'][2]+'positive_accuracy',
                 util_fns.calculate_positive_accuracy(positive_preds_dict),batch_size=self.batch_size)
        self.log(self.metrics['split'][2]+'positive_accuracy_by_fire',
                 util_fns.calculate_positive_accuracy_by_fire(positive_preds_dict),batch_size=self.batch_size)

        # Use 'global_step' to graph positive_accuracy_by_time and positive_cumulative_accuracy
        positive_accuracy_by_time, positive_cumulative_accuracy = util_fns.calculate_positive_accuracy_by_time(positive_preds_dict)

        for i in range(len(positive_accuracy_by_time)):
            self.logger.experiment.add_scalar(self.metrics['split'][2]+'positive_accuracy_by_time',
                                             positive_accuracy_by_time[i], global_step=i)
            self.logger.experiment.add_scalar(self.metrics['split'][2]+'positive_cumulative_accuracy',
                                             positive_cumulative_accuracy[i], global_step=i)

        average_time_to_detection, median_time_to_detection, std_time_to_detection = util_fns.calculate_time_to_detection_stats(positive_preds_dict, positive_times_dict)
        # self.log(self.metrics['split'][2]+'average_time_to_detection', average_time_to_detection,batch_size=self.batch_size)
        # self.log(self.metrics['split'][2]+'median_time_to_detection', median_time_to_detection,batch_size=self.batch_size)
        # self.log(self.metrics['split'][2]+'std_time_to_detection', std_time_to_detection,batch_size=self.batch_size)
        
        print("Computing Test Evaluation Metrics Complete.")


if __name__ == "__main__":

    image_data_path="frame"
    prev_image_data_path="prev_frame"
    labels_path="bbox_annotation"
    optical_flow_path="opt_flow_frame"

    train_split_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/train"
    val_split_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/val"
    test_split_path="/home/natalia/smoke_workspace/grid_module/data/image_dataset/test"

    data_module = SmokeDataModule(
        image_data_path=image_data_path,
        prev_image_data_path=prev_image_data_path,
        labels_path=labels_path,
        optical_flow_path=optical_flow_path,

        train_split_path=train_split_path,
        val_split_path=val_split_path,
        test_split_path=test_split_path,

        batch_size=2, 
        num_workers=0, 
        
        original_dimensions = (480, 640),
        resize_dimensions = (480, 640),
        n_tile_size = (10,10)
    )


    batch_size=1
    num_workers=0
    series_length=2
    add_base_flow=False
    time_range=(-2400,2400)

    original_dimensions=(1536, 2016)
    resize_dimensions=(1536, 2016)
    crop_height=1120

    tile_dimensions=(224,224)
    tile_overlap=0
    pre_tile=True
    smoke_threshold=250
    num_tile_samples=0

    flip_augment=True
    resize_crop_augment=True
    blur_augment=True
    color_augment=True
    brightness_contrast_augment=True

    # Model args
    model_type_list=['RawToTile_MobileNet']
    pretrain_epochs=None
    intermediate_supervision=True
    use_image_preds=False
    tile_embedding_size=960

    pretrain_backbone=True
    freeze_backbone=False
    backbone_size='small'
    backbone_checkpoint_path=None

    tile_loss_type='bce'
    bce_pos_weight=36
    focal_alpha=0.25
    focal_gamma=2
    image_loss_only=False
    image_pos_weight=1
    confidence_threshold=0

    # Optimizer args
    optimizer_type='SGD'
    optimizer_weight_decay=0.0001
    learning_rate=0.0001
    lr_schedule=True

    # Trainer args 
    min_epochs=3
    max_epochs=50
    early_stopping=True
    early_stopping_patience=4
    sixteen_bit=True
    stochastic_weight_avg=True
    gradient_clip_val=0
    accumulate_grad_batches=1


    main_model = MainModel(
                        # Model args
                        model_type_list=['RawToTile_MobileNet',"TileToTile_Transformer","TileToTileImage_SpatialViT"],
                        pretrain_epochs=None,
                        error_as_eval_loss=None,
                        use_image_preds=None,
                        tile_embedding_size=None,
                        num_tiles=100,
                        num_tiles_height=10,
                        num_tiles_width=10,
                        series_length=2)
    print("opt", optimizer_type)
    lightning_module = LightningModule(
                               model=main_model,
                               batch_size=batch_size,

                               optimizer_type=optimizer_type,
                               optimizer_weight_decay=optimizer_weight_decay,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,

                               series_length=series_length)

    ### Implement EarlyStopping & Other Callbacks ###
    callbacks = []
    early_stop_callback = EarlyStopping(
                            monitor='train/tile_f1',
                            min_delta=0.00,
                            patience=early_stopping_patience,
                            verbose=True)
    callbacks.append(early_stop_callback)
    checkpoint_callback = ModelCheckpoint(monitor='train/tile_f1', save_last=True)
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)


    trainer = pl.Trainer(
        # Trainer args
        # min_epochs=min_epochs,
        # max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=512,
        gpus=1)

    ### Training & Evaluation ###
    # if is_test_only:
    #     trainer.test(lightning_module, datamodule=data_module)
    # else:
    #     trainer.fit(lightning_module, datamodule=data_module)
    #     trainer.test(lightning_module, datamodule=data_module)
    trainer.fit(lightning_module, datamodule=data_module)
    # trainer.test(lightning_module, datamodule=data_module)