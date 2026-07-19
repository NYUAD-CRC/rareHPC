# rareHPC

`rareHPC` is a project for running the test pipeline using a preconfigured Conda environment and pre-trained machine learning models.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/NYUAD-CRC/rareHPC.git
   cd rareHPC
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate rareHPC
   ```

4. Install the required pip dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Download the pre-trained models:

   ```bash
   python download_models.py
   ```

## Running the Test Pipeline

Run the test pipeline using:

```bash
python test.py --y "memory_efficiency_%"
```
This gives the results on test data for memory efficiency.

```bash
python test.py --y "%_UtilisedTime"
```
This gives the results on test data for compute time utilisation.
## Project Structure

```text
riskHPC/
├── environment.yml
├── requirements.txt
├── download_models.py
├── test.py                           # script used to test the trained models on test data
├── train.py                          # script used to train the machine learning models
├── DataPreprocess.ipynb              # jupyternotebook used to clean the data and curate features
├── ComputeTimewithSampling/          # Model files and Results for compute time utilisation
├── MEMwithsampling/                  # Model files and Results for memory utilisation
├── classification_model/             # Model files Results for High/Low compute time utilisation
└── ...
```

## Notes

* The `download_models.py` script downloads the required models and places them in the appropriate location.
* Ensure the model download completes successfully before running `test.py`.
* The test data is in the file test.csv on which results are generated. 
