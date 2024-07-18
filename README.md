# WadTCN
To create the environment needed to test 2 Andi Challenge results, please execute the following line:

    conda env create -f environment-cpu.yml

All `2_andi_challenge*.py` scripts are the ones used to train and infer 2nd Andi Challenge networks and datasets, respectively. Please, execute the scripts in the main folder of the repository, not inside scripts directory.

To reproduce the exact best submission for the challenge, follow these steps:

  
1. Download the directories from
    https://drive.google.com/drive/folders/1y2pCye_tef21a3QKDS22ZPUMDHLQtCvj?usp=sharing.

2. Place these directories inside the main directory of your GitHub repository.

3. Execute the script '2nd_andi_challenge_submission.py' in the main repository directory.
    
Results will be generated and stored in a directory named '2nd_andi_challenge_results'. The ensemble task requires some visual assistance from the user, so minimal supervision is necessary. You'll need to input the number of "peaks" observed in the distributions. These steps should help you replicate the best submission for the challenge effectively. For exact ensemble results replication, use the following number of "peaks":

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

