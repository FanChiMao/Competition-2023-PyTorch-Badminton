# [AICUP 2023] Competition-2023-Pytorch-Badminton

## TEAM_2970: [Jonathan](https://github.com/FanChiMao), Joe, Dodo, Edward, [Harry](https://github.com/SHRHarry)  

- [**[AICUP 2023] Shuttlecock Recognization**](https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8)  

<a href="https://aidea-web.tw/topic/cbea66cc-a993-4be8-933d-1aa9779001f8"><img src="https://i.imgur.com/rwfx1h8.png" title="source: top image" /></a>  

[![report](https://img.shields.io/badge/Supplementary-Report-yellow)](https://drive.google.com/file/d/1bBWmC9laZLcsNunM-4R0B0-r_z41eIDL/view?usp=sharing) 
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
- Public Training data: https://drive.google.com/drive/folders/16fJ4fKBQs-xvJI8jGq02k0MnLFAIcg30?usp=sharing  
- Public Testing data: https://drive.google.com/drive/folders/1_k5u7bnnPa890a_Mc9DHpU2lbQkPFR28?usp=drive_link  
- Private Testing data: https://drive.google.com/drive/folders/1-seTiss3bGpF9T2tHQofQV4BKEKfaV71?usp=drive_link  


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
- Chi-Mao Fan: qaz5517359@gmail.com  
- Hong-Ru Shen: ms024929548@gmail.com  
`TODO`
