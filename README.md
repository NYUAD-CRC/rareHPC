# riskHPC

`riskHPC` is a project for running the test pipeline using a preconfigured Conda environment and pre-trained machine learning models.


## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd riskHPC
   ```

2. Create the Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate riskHPC
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
python test.py
```

## Project Structure

```text
riskHPC/
├── environment.yml
├── requirements.txt
├── download_models.py
├── test.py
├── ComputeTimewithSampling/          # Model files and Results for compute time utilisation
├── MEMwithsampling/                  # Model files and Results for memory utilisation
├── classification_model/             # Model files Results for High/Low compute time utilisation
└── ...
```

## Notes

* The `download_models.py` script downloads the required models and places them in the appropriate location.
* Ensure the model download completes successfully before running `test.py`.
* The test data is in the file test.csv on which results are generated. 
