# Swin Transformer Model for Image Classification

This project implements an image classification model using the Swin Transformer architecture. The model is designed to handle an imbalanced dataset by utilizing class-weighted loss, and it is trained using the PyTorch framework.

## Project Structure
- Training and Validation Data: The dataset is loaded from folders containing images classified into different categories.
- Model: A pre-trained Swin Transformer model (swin_base_patch4_window7_224) is fine-tuned for the specific classification task.
- Loss Function: The class imbalance is handled by applying class weights to the cross-entropy loss.
- Optimizer: The model is optimized using AdamW.

## Requirements
To install the required libraries, run:
``` bash
pip install torch torchvision timm scikit-learn tqdm pandas
```
## Dataset 
The training and validation dataset has been developed using
three publicly available (SEE-AI project dataset, KID,
and Kvasir-Capsule dataset) and one private dataset (AIIMS) VCE datasets. The training and validation dataset
consist of 37,607 and 16,132 VCE frames respectively mapped to 10 class labels namely angioectasia, bleeding, erosion, erythema, foreign body, lymphangiectasia, polyp, ulcer, worms,
and normal.
| Type of Data | Source Dataset | Angioectasia | Bleeding | Erosion | Erythema | Foreign Body | Lymphangiectasia | Normal | Polyp | Ulcer | Worms |
|--------------|----------------|--------------|----------|---------|----------|---------------|------------------|--------|-------|-------|-------|
| Training     | KID            | 18           | 3        | 0       | 0        | 0             | 6                | 315    | 34    | 0     | 0     |
|              | KVASIR         | 606          | 312      | 354     | 111      | 543           | 414              | 24036  | 38    | 597   | 0     |
|              | SEE-AI         | 530          | 519      | 2340    | 580      | 249           | 376              | 4312   | 1090  | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 66    | 158   |
| **Total Frames** |                | **1154**     | **834**  | **2694**| **691**  | **792**       | **796**          | **28663**| **1162**| **663**| **158** |
| Validation   | KID            | 9            | 2        | 0       | 0        | 0             | 3                | 136    | 15    | 0     | 0     |
|              | KVASIR         | 260          | 134      | 152     | 48       | 233           | 178              | 10302  | 17    | 257   | 0     |
|              | SEE-AI         | 228          | 223      | 1003    | 249      | 107           | 162              | 1849   | 468   | 0     | 0     |
|              | AIIMS          | 0            | 0        | 0       | 0        | 0             | 0                | 0      | 0     | 29    | 68    |
| **Total Frames** |                | **497**      | **359**  | **1155**| **297**  | **340**       | **343**          | **12287**| **500** | **286** | **68** |

### Dataset Structure
The images are organized into their respective classes for both the training and validation datasets as shown below:
```bash
Dataset/
├── training
│   ├── Angioectasia
│   ├── Bleeding
│   ├── Erosion
│   ├── Erythema
│   ├── Foreign Body
│   ├── Lymphangiectasia
│   ├── Normal
│   ├── Polyp
│   ├── Ulcer
│   └── Worms
│   └── training_data.xlsx
└── validation
    ├── Angioectasia
    ├── Bleeding
    ├── Erosion
    ├── Erythema
    ├── Foreign Body
    ├── Lymphangiectasia
    ├── Normal
    ├── Polyp
    ├── Ulcer
    └── Worms
    └── validation_data.xlsx
```

You can update the dataset path in the code as needed:
```
train_dataset = ImageFolder(root='/path/to/training', transform=train_transform)
val_dataset = ImageFolder(root='/path/to/validation', transform=val_transform)
```

## Training the Model
The model is trained for a fixed number of epochs, with data augmentation applied to the training set. You can run the training using the following code structure:



```bash
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
```

## Evaluation
After training, the model's performance is evaluated using the balanced accuracy score to account for the imbalanced dataset.

```
evaluate(model, val_loader)
```

## Prediction on Test Data
To generate predictions on the test dataset, the following function is used. It outputs a CSV file containing the predicted class probabilities and the predicted labels for each image.


```
predict(model, test_loader)
```

## Model Components
- Data Augmentation: Techniques like random horizontal flipping, rotations, resizing, and color jittering are applied to augment the training data.
- Swin Transformer: A transformer-based architecture that processes images in patches, focusing on hierarchical feature learning.
- Class-Weighted Loss: To address the imbalanced dataset, class weights are computed and passed to the cross-entropy loss function.
- Optimizer: AdamW optimizer is used, which is known for its weight decay regularization properties.

## Results
The performance of the model is evaluated based on the Balanced Accuracy metric, which is particularly useful for imbalanced datasets.

## References
- Swin Transformer Paper
- PyTorch Documentation
- timm - PyTorch Image Models
