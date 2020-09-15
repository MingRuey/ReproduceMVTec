# ReproduceMVTec
Reproduce the CVPR2020 paper: [Uninformed Students: Student-Teacher Anomaly Detection With Discriminative Latent Embeddings](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.pdf)

&& this: [Modeling the Distribution of Normal Data in Pre-Trained Deep Features for Anomaly Detection](https://arxiv.org/abs/2005.14140)


### Dependency:
    tensorflow >= 2.3

### ToDo:
    - Understand and program the distillation loss L_k
    - Understand and program the triplet loss L_m
    - Understand and program the compactness loss L_c
    - Program a crop/sample preprocess pipe, incorporate above losses

### Finished:
    - Prepare MVTec dataset
    - Prepare Resnet-18 pretrained (Use keras pretrained Res50V2 instead)
    - Construct the teacher/student network
