# Uncertainty-Aware Artery/Vein Classification

![](https://raw.githubusercontent.com/agaldran/a_v_uncertain/master/utils/predictions_example.png?style=centerme)
<p align="center">
(a)-(e):  Retinal images extracted from different dataset: (a) RITE/DRIVE (b) INSPIRE (c) HRF (d) STARE (e) LES-AV . 
(f)-(j): Corresponding predictions.
Note that, in contrast to most existing technques, this model was trained in RITE and never re-trained, see paper for details.
<p align="center">

## DESCRIPTION

This is an implementation of a method for Artery/Vein segmentation with uncertainty predictions.
If you find this code useful for your research, we would appreciate if you could cite:
```
Uncertainty-Aware Artery/vein Classification on Retinal Images
A. Galdran,  M. I. Meyer, P. Costa, A. M. Mendonça, A. Campilho
IEEE International Symposium on Biomedical Imaging (ISBI), 2019
```
* **PDF**: Follow this [link](https://agaldran.github.io/pdf/uncertainty_aware_av.pdf)

## INSTRUCTIONS
Assuming you have cloned this repository to a working folder `F/` already have a working Anaconda distribution installed in your system, you just need to execute the three lines below:

```
conda create --name av_uncertain python=3.7
source activate av_uncertain
conda install --file requirements.txt
```

And you are almost ready to go. After this, just `cd` into `F/`  and first download the weights to the `models/checkpoints_uncertainty/` subfolder:
```
wget https://gitlab.com/agaldran/shared_models/raw/9ebce839b046a115ff7ba5defc6251a139eedfda/model_final.pth.tar -P models/checkpoints_uncertainty/
```
Once finished, run:

```
python build_predictions.py --path_ims retinal_images --path_out results
```
where `retinal_images` is the path containing the images you want to generate predictions for, and `results` can be replaced by the location where you want your results stored.

## INTERPRETING THE RESULTS
Note that this method provides pixel-wise predictions divided into four different classes: Background, Artery, Vein, and Uncertain. 
The results will come out color-coded:
* Red: Probability of being an Artery pixel
* Blue: Probability of being a Vein pixel
* Green: Probability of being an Uncertain prediction
* Background: 1-Artery-Vein-Uncertain

Running the above line will generate three subfolders of ``results/``: ``uncertainty``, ``uncertainty_vessels``, and ``pretty_preds``. 
* Inside the ``uncertainty/`` folder you will find predictions color-coded as above. 
* Inside the ``uncertainty_vessels/`` folder, Artery/Vein/Uncertainty have been merged into a single class (vessels), thereby providing a vessel segmentation model.
* Inside the ``pretty_preds`` folder, background predictions are reverted, so that pixels predicted as background will show up close to white, which improves predictions legibility, but this is only an aesthetic opinion of mine :)

## TRAINING YOUR MODEL
Code for training your own model will come soon (I hope).



