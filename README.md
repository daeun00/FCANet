# Low-Light Image Enhancement via Wavelet Domain Frequency Cross-Attention 

## Abstract
This study proposes a novel low-light image enhancement network that incorporates a frequency
cross-attention mechanism in the wavelet domain. The proposed network enhances
brightness through the low-frequency wavelet subband while simultaneously restoring fine
details in the high-frequency subbands. Color degradation during the brightening process
is prevented by applying a color-preserving block based on the saturation component
before the illumination adjustment. Furthermore, U-shaped lightening and multiscale
sharpening blocks are designed to enhance the image brightness and detail, respectively.
The disruption of intrinsic symmetry in coefficient correlations poses a major challenge
in independently processing wavelet subbands. To address this issue, we propose a frequency
cross-attention block that enables effective information exchange between subbands,
thereby preserving their inherent correlations. The proposed network produces visually
consistent and refined outputs by balancing the enhanced wavelet subbands. Experimental
evaluations demonstrate that the proposed network achieves competitive performance
in both subjective quality and objective metrics, confirming its effectiveness for low-light
image enhancement. :book: : [[Paper]](https://doi.org/10.3390/sym18030470)

## Architecture
* **Overall Architecture**<br/>
![Alt text](/figures/overall_architecture.png)
<br/>

## Requirements
* Python 3.8
* PyTorch 2.4
* CUDA 11.8
```
pip install pillow, opencv-python, scikit-image, sacred, pymongo
```

## Test
* Put test images under *./test_img* folder.
* Put the trained model under  *./models* folder.
<br/> - You can download best model file [here](https://drive.google.com/drive/folders/1jZIQUwlCCqy2MS9X4dBtfljbFTZX6hGi?usp=drive_link)
* Run test.py
```
python test.py --modelfile models/FCANet.pth
```
* The test results will be saved to the folder: ./output.
<br/>

## Results
* You can check example results about *DICM*, *Fusion*, *LIME*, *LOL*, *MEF*, *VV* datasets in each folder.<br/>
* **Qualitative Comparison**<br/>
<img src="/figures/qualitative_comparision.png" width="70%" height="70%" title="Qualitative Comparison"></img><br/>
<br/>


## Citation Information
```
@article{FCANet2026lee,
  title={Low-Light Image Enhancement via Wavelet Domain Frequency Cross-Attention},
  author={Lee Da Eun, Jun Young Park, and Il Kyu Eom},
  journal={Symmetry},
  year={2026},
}
```  
