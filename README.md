# ktb_recommendation

requirement.txt
1. dgl == version2.00
2. torch==2.5.1 torchaudio==2.5.1 torchtext==0.14.1torchvision==0.20.1
3. python = 3.10
4. dask==2024.10.0

# Project Structure

This project contains the following directory structure:

```
📁 input  
├── Creator_random.csv            # CSV file containing creator (user) data  
└── Item_random.csv               # CSV file containing item (proposal) data  

📁 output  
├── data.pkl                      # Serialized data object for quick loading  
├── item_embedding.pth            # Embedding weights for item data  
├── item_embeddings.pth           # Alternative item embedding file  
├── saved_model.pth               # Saved model file for inference or transfer learning  
├── train_g.bin                   # Binary graph data file for training  
└── trained_model.pth             # Final trained model weights  

📁 test  
├── test_layers.py                # Unit tests for layers used in the model  
├── test_model_recommend.py       # Unit tests for the recommendation model  
├── test_sampler.py               # Unit tests for data sampling functions 
              
               
├── .gitignore                    # Git ignore file to exclude unnecessary files  
├── builder.py                    # Code file for building the model or graph structures  
├── data_utils.py                 # Utilities for data preprocessing and management  
├── evaluation.py                 # Evaluation metrics and functions for model performance  
├── layer_origin.py               # Contains the original layer configurations  
├── layers.py                     # Custom layer definitions used in the model  
├── main.py                       # Main execution script for training or running the model  
├── model_recommend.py            # Recommendation model definition and training logic  
├── process_data.py               # Script for processing raw data into model-compatible format  
├── README.md                     # Project documentation (this file)  
├── recommend.py                  # Code for generating recommendations using the trained model  
├── requirements.txt              # Package dependencies for the project  
├── requirements_test.txt         # Additional dependencies for testing  
├── requirments.txt               # Possibly a duplicate or misspelled requirements file  
└── sampler.py                    # Sampler functions to handle data sampling for training  
```

b
