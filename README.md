# Competition-2023-Pytorch-Badminton

## TEAM_2970: 

- [**[AICUP 2023] Shuttlecock Recognization**](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)  

<a href="https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8"><img src="https://i.imgur.com/rwfx1h8.png" title="source: top image" /></a>  

[![report](https://img.shields.io/badge/Supplementary-Report-yellow)]() 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() 
![Visitor Count](https://visitor-badge.glitch.me/badge?page_id=Competition-2023-PyTorch-Badminton)

## Demo result  


## Installation  
- Clone the code from repository  
    ```
    git clone https://github.com/FanChiMao/Competition-2023-PyTorch-Badminton
    ```


- Build the environment
    ```
    cd Competition-2023-PyTorch-Badminto
    pip install -r requirements.txt
    ```

- Download the TrackNetv2
    ```
    git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2

    python utils\predict_process\TrackNet_import.py
    ```

## Dataset  
`TODO`


## Inference 
- Download our YOLOv8 trained weights by following commands
    ```
    cd trained_weights
    python download_trained_weights.py
    ```
    Or you can directly download from [**here**]().`TODO`  


- To predict the video input, run `TODO`
    ```
    python main_predict.py --video {video_path}  
    ```

## Reference  
- https://github.com/ultralytics/ultralytics
- https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2



## Contact us  
- Chi-Mao Fan (leader): qaz5517359@gmail.com  
`TODO`
