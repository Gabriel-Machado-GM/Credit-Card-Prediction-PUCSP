# Credit Card Prediction PUCSP 2024.2 Project

## Project Objective
The objective of this project is to predict the likelihood of default payments by credit card clients using a dataset from Kaggle. This dataset contains detailed information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

## Dataset
This dataset is sourced from the Kaggle link: [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).

### Dataset Information
The dataset contains 25 variables, including:
- **ID**: ID of each client
- **LIMIT_BAL**: Amount of given credit in NT dollars (includes individual and family/supplementary credit)
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Repayment status from April 2005 to September 2005
- **BILL_AMT1 to BILL_AMT6**: Amount of bill statement from April 2005 to September 2005
- **PAY_AMT1 to PAY_AMT6**: Amount of previous payment from April 2005 to September 2005
- **default.payment.next.month**: Default payment (1=yes, 0=no)

Inspiration for exploration includes analyzing how the probability of default payment varies by different demographic variables and identifying the strongest predictors of default payment.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To run the project, run the .ipynb archive.

## Contribution
To contribute to this project, follow these steps:
1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

For more details, refer to the [README.md](https://github.com/Gabriel-Machado-GM/Credit-Card-Prediction-PUCSP-2024.2-/blob/0fae35b42975f182f228aa045ce338b76111bd25/README.md) in the repository.
