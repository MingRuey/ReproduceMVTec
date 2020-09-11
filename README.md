# ReproduceMVTec
Reproduce the CVPR2020 paper:  Uninformed Students: Student-Teacher Anomaly Detection With Discriminative Latent Embeddings


### Dependency:
    tensorflow >= 2.3

### ToDo:
    - Prepare Resnet-18 pretrained
    - Construct the teacher/student network
    - Understand and program the distillation loss L_k
    - Understand and program the triplet loss L_m
    - Understand and program the compactness loss L_c
    - Program a crop/sample preprocess pipe, incorporate above losses

### Finished:
    - Prepare MVTec dataset
