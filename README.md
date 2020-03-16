# Starbucks Offer Engine

The project is about creating an offer recommendation engine for the Starbucks
marketing team. Every few days, Starbucks is sending out offers to users of the
mobile app. There are mainly three main types of offers: discount, BOGO (buy one get one) and informational offers. Not all Starbucks customers receive the same offer. 

The problem of the project is to find a solution that will take customer
attributes and offer attributes into account and suggest if the offer will be
successful or not. By doing so Starbucks team will be able to check to which
offer a customer will respond. By finding the right offer for the right customer
Starbucks will be able to increase the marketing ROI. Also, Starbucks will be
able to target only customers that will respond to the offers, instead of
sending the same offer to everyone and customers may become annoyed by
Starbucks's advertising.


## Pre-requisites:
- Python v3.8
- Scikit-learn 0.22.2
- Pandas v1.0.1
- Numpy v1.18.0
- XGBoost v1.0.2
- Vecstack v0.4.0
- Seaborn v0.10.0
- Matplotlib v3.2.0

## Getting Started

The whole project is decomposed into four parts: preprocessing, feature engineering, training, and prediction. Each part runs independently from each other. EDA is provided in the jupyter notebook file under `./docs/submission/starbucks-notebook.ipynb`. The description of EDA is in the final report, you can find the draft version here `./docs/submission/report-draft.pdf`.

* To start the preprocessing part, use `./run preprocess`
* To start the feature engineering part, use `./run features`
* To start the training part, use `./run train`
* To start the prediction part, use `./run predict`

## Support 

Patches are encouraged and may be submitted by forking this project and submitting a pull request through GitHub.

## Credits

The project was developed during the ML program of
[Udacity.com](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t)

## Licence

Released under the [MIT License](./License.md)
