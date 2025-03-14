# WadTCN

To create the environment needed to test 2 Andi Challenge results, please execute the following line with conda:

    conda env create -f environment-cpu.yml

All `scripts/2_andi_challenge*.py` scripts are the ones used to train and infer 2nd Andi Challenge networks and datasets, respectively. Please, execute the scripts in the main folder of the repository, not inside scripts directory.

To reproduce the exact best submission for the challenge, follow these steps:

1. (Optional) Download and unzip [OneDrive Link](https://ucao365-my.sharepoint.com/:f:/g/personal/francisco_barrantes_uca_edu_ar/EjV0y0OHd2tPg_-b9CONFD0B6PuRrCebG2XHZl7fWdCQ7A?e=bj5vCs). You can ignore `public_data_challenge_v0` but we recommend you to use that directory instead the one provided in the challenge. The reason for this is that the provided that directory includes `.csv` files including extracted trajectories from videos using UNet and trackpy. If you do not want to use it, execute first `2nd_andi_challenge_extract_trajectories_from_videos.py` and then `2nd_andi_challenge_submission.py`.
2. Place this directory inside the main directory of your GitHub repository.
3. Execute the script `2nd_andi_challenge_submission.py` in the main repository directory.
4. Results are found in the created directory `2nd_andi_challenge_results`.

The ensemble task requires some visual assistance from the user, so user supervision is necessary. You'll need to input the number of "peaks" observed in the distributions. These steps should help you replicate the best submission for the challenge effectively. For exact ensemble results replication, use the following number of "peaks":

|Number of Experiment|Number of peaks - Track 1|Number of peaks - Track 2|
|:----|:----|:----|
|0|1|1|
|1|1|1|
|2|2|2|
|3|2|2|
|4|2|2|
|5|1|1|
|6|2|1|
|7|2|2|
|8|1|2|
|9|3|2|
|10|1|1|
|11|2|2|

To delete the environment type:

    conda remove --name anomalous_diffusion --all
