# [AICUP 2023] Competition-2023-Pytorch-Badminton

## TEAM_2970: [Jonathan](https://github.com/FanChiMao), Joe, Dodo, Edward, [Harry](https://github.com/SHRHarry)  

- [**[AICUP 2023] Shuttlecock Recognization**](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)  

<a href="https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8" target="_blank"><img src="https://i.imgur.com/rwfx1h8.png" title="source: top image" /></a>  

[![report](https://img.shields.io/badge/Supplementary-Report-yellow)](https://drive.google.com/file/d/1bBWmC9laZLcsNunM-4R0B0-r_z41eIDL/view?usp=sharing)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FFanChiMao%2FCompetition-2023-Pytorch-Badminton&label=visitors&countColor=%232ccce4&style=plastic)

## Model architectures

<table xmlns="http://www.w3.org/1999/html">
  <tr>
    <td colspan="3"><img src = "https://i.imgur.com/VxSoOm7.png" alt="model" width="900"> </td>  
  </tr>
  <tr>
    <td colspan="3"><p align="center"><b>Overall pipeline</b></p></td>
  </tr>
  
  <tr>
    <td> <a href="https://arxiv.org/abs/2212.04500" target="_blank"><img src = "https://i.imgur.com/arDSL6L.png" width="300"></a> </td>
    <td> <a href="https://ieeexplore.ieee.org/document/9275314" target="_blank"><img src = "https://i.imgur.com/KrS4cta.png" width="300"></a> </td>
    <td> <a href="https://ieeexplore.ieee.org/document/9302757" target="_blank"><img src = "https://i.imgur.com/hkrGkDe.png" width="300"></a> </td>
  </tr>
  <tr>
    <td><a href="https://github.com/ruiwang2021/mvd" target="_blank"><p align="center"><b>MVD<br>(Video Distillation)</b></p></a></td>
    <td><a href="https://github.com/li-plus/DSNet" target="_blank"><p align="center"> <b>DSNet<br>(Video Summarization)</b></p></a></td>
    <td><a href="https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2" target="_blank"><p align="center"> <b>TrackNetv2<br>(Video Segmentation)</b></p></a></td>
  </tr>
</table>

## Each model purpose
- MVD (to predict the ball type)
- DSNet (to predict the shot frame number)
- TrackNetv2 (to predict the shuttlecock position, ball height)
- YOLOv8-Detection (to predict the player position)
- YOLOv8-PoseEstimation (to predict the toe position)
- YOLOv8-Classification (to predict the round head, backhand)

## Installation  
- Clone the code from repository  
    ```
    git clone https://github.com/FanChiMao/Competition-2023-PyTorch-Badminton
    ```
- Install submodule
  ```
  cd Competition-2023-PyTorch-Badminton
  git submodule update --init
  ```

- Build the environment
    ```
    cd Competition-2023-PyTorch-Badminton
    pip install -r requirements.txt
    ```

- Download the TrackNetv2
    ```
    git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
    python utils\predict_process\TrackNet_import.py
    ```

## Datasets  
- [**Public Training data**](https://drive.google.com/drive/folders/16fJ4fKBQs-xvJI8jGq02k0MnLFAIcg30?usp=sharing) 
- [**Public Testing data**](https://drive.google.com/drive/folders/1_k5u7bnnPa890a_Mc9DHpU2lbQkPFR28?usp=drive_link)  
- [**Private Testing data**](https://drive.google.com/drive/folders/1-seTiss3bGpF9T2tHQofQV4BKEKfaV71?usp=drive_link)  


## Inference 
- Download our YOLOv8 trained weights by following commands, or you can directly download from [**here**](https://github.com/FanChiMao/Competition-2023-PyTorch-Badminton/releases/tag/v0.0).

    ```
    cd trained_weights
    python download_trained_weights.py
    ```
    
- Check the configuration path from `./inference.yaml`
    ```
    # Path setting
    PATH:
      VIDEO: D:\AICUP\datasets\test\video
      HIT_CSV: D:\AICUP\datasets\test\predict_csv
      RESULT: .\predict_result
      OPENPOSE: .\for_openpose
    
    # Pretrained weights path
    WEIGHTS:
      PLAYER: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s-players_detection_2.pt
      COURT: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s-seg_net_detection.pt
      NET: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s-players_detection.pt
      ROUNDHEAD: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8n-cls_roundhead.pt
      BACKHAND: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8n-cls_backhand.pt
      BALLTYPE: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s-cls_balltypes.pt
      START: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8n-cls_balltypes_start.pt
      AFTER: D:\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8n-cls_balltypes_after.pt
    ```
  
- After setting the configuration, to predict the video input, simply run 
    ```
    python main_predict.py
    ```

## Reference  
- https://github.com/ultralytics/ultralytics
- https://github.com/ruiwang2021/mvd
- https://github.com/li-plus/DSNet
- https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2


## Contact us  
- Chi-Mao Fan: qaz5517359@gmail.com  
- Hong-Ru Shen: ms024929548@gmail.com  

