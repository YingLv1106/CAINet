# CAINet 
  This project provides the code and results for Context-Aware Interaction Network for RGB-T Semantic Segmentation", IEEE TMM, 2023. [IEEE link](https://iee) and [[arxiv link]](h) [[Homepage](https://github.com/YingLv1106/CAINet)]

# Requirements
  python 3.7 + pytorch 1.12.0

# Network

   <div align=center>
   <img src="https://github.com/YingLv1106/CAINet/blob/main/image/cainet.png">
   </div> 

# Segmentation maps and performance

   We provide segmentation maps on MFNet dataset and PST900 dataset [[GoogleDrive](https://drive.google.com/drive/folders/1fKE9JpyhLWPIzHaqSJ4zl7t0WafdqYkI?usp=drive_link)] [[BaiDu]](https://pan.baidu.com/s/1Z0zEw527UTtCccKtyTznww?pwd=arn3) (arn3)

   **Performace on MFNet dataset**

   <div align=center>
   <img src="https://github.com/YingLv1106/CAINet/blob/main/image/resul_mfnet.jpg">
   </div>

   **Performace on PST900 dataset**

   <div align=center>
   <img src="https://github.com/YingLv1106/CAINet/blob/main/image/resul_pst.jpg">
   </div>


# Pre-trained model and testing
1. Download the following pre-trained model and put it under './checkpoint' [[download checkpoint GoogleDrive](https://drive.google.com/drive/folders/1dX7frPekYnw1nR9rwnbwZkcdC6Zj2gLT?usp=drive_link)]  [[BaiDu]](https://pan.baidu.com/s/1Z0zEw527UTtCccKtyTznww?pwd=arn3) (arn3)
2. run evaluate_*.py.


# Citation
    @ARTICLE{lv2023cainet,
      author={Lv, Ying and Liu, Zhi and Li, Gongyang},
      title={Context-Aware Interaction Network for RGB-T Semantic Segmentation}, 
      journal={IEEE Transactions on Multimedia}, 
      volume={},
      number={}, 
      year={2023},
      pages={1-13},
      doi={10.1109/TMM.2023.3349072}
      }
    
