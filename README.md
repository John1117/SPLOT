# SPLOT - Sport Lottery Analysis
In this project, we are going to analyze expected return of playing sport lottery. We primarily focused on soccer games. If expected return is once positive, it is worth playing. This project includes three models along with their corresponding research phrases: 
1. **Odd Shifting Model**: Originally, we wanted to predict the true winning odd shifted from those odds provided by several online sport lottery retailers. 
2. **Max Exploit Model**: Further, we wanted to determine the optimal retailer's odd by maximally exploiting players from a retailer's perspective. 
3. **Poisson Model**: Finally, we found that a scoring distribution of a soccer game follows Poisson distribution. We could depict this distribution from its historical scoring data.

Once we find a positive expected return of playing a certain game, we then apply the Kelly's Criterion to detemine a betting size out of our total funding.

## Finished
1. `odd_shifting_model.py`: The implementation of Odd Shifting Model.
2. `max_exploit_model.py`: The implementation of Max Exploit Model.

## TODO
1. Poisson Model
2. Kelly's Criterion

## Getting Start

### Prerequisites
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/John1117/SPLOT.git
    ```
    
2. Install the dependencies using:
    ```bash
    cd SPLOT
    pip install -r requirements.txt