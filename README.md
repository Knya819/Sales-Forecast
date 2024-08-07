# Sales Forecasting Project

## Overview
The Sales Forecasting Project is designed to predict the future sales quantities of products using historical sales data. By leveraging machine learning techniques, specifically LSTM (Long Short-Term Memory) networks, this project aims to provide accurate forecasts to help businesses manage inventory and optimize their sales strategies.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data Description](#data-description)
5. [Model Description](#model-description)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## Project Structure
- `Dataset.csv`: The dataset containing historical sales data.
- `Sales.py`: Script for sales forecasting without additional parameters.
- `Sales_Coupon.py`: Script for sales forecasting considering coupons.
- `Sales_Return.py`: Script for sales forecasting considering returns.
- `model.py`: Script for sales forecasting considering both coupons and returns.
- `README.md`: This file.
- `requirements.txt`: List of Python packages required for the project.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Sales-Forecast.git


## Usage
Each script is designed to forecast sales based on different parameters. You can run any of these scripts depending on the requirement:

To run the script without additional parameters:

```bash
python Sales.py
```

To run the script considering coupons:

```bash
python Sales_Coupon.py
```

To run the script considering returns:
```bash
python Sales_Return.py
```
To run the script considering both coupons and returns:
```bash
python model.py
```

## Data Description

The dataset (`Dataset.csv`) contains the following columns:

- **item_name**: Name of the item.
- **item_brand**: Brand of the item.
- **item_main_category**: Main category of the item.
- **item_sub_category**: Sub-category of the item.
- **transaction_date**: Date of the transaction.
- **item_quantity**: Quantity of items sold.
- **return_quantity**: Quantity of items returned (used in `Sales_Return.py` and `model.py`).
- **item_coupon**: Coupon applied on the item (used in `Sales_Coupon.py` and `model.py`).

## Model Description

The model used in this project is an LSTM (Long Short-Term Memory) network. The architecture includes:

- Two LSTM layers with ReLU activation.
- Dropout layer to prevent overfitting.
- Dense layer to output the prediction.

## Results

The model's predictions are visualized and compared with the original sales data to assess its accuracy. The forecasted sales quantities help in understanding future sales trends.

## Contributing

Feel free to open issues or submit pull requests if you have any improvements or suggestions.


## License
This project is licensed under the MIT License.


