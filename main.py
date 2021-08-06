import streamlit as st
import scipy.optimize
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class ExponentialClassifier(object):
    def __init__(self, discountRate=-5, rho=.01):
        self.discountRate = discountRate
        self.rho = rho

    def fit(self, X, y):
        ranges = [[-8, .01, .1], [0, 1.1, .1]]
        opt = scipy.optimize.brute(self.sse, ranges, args=(X, y,), finish=scipy.optimize.fmin)
        self.discountRate = opt[0]
        self.rho = opt[1]
        return self

    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > -.01) or (params[1] <= 0.):
            return 100000000000000000000000 #return something absurd
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat_clip = np.clip(yhat, boundary, 1 - boundary)
        asdf1 = -y * np.log(yhat_clip)
        asdf2 = (1 - y) * np.log(1 - yhat_clip)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        # print([params, sse])
        return sse

    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
        ssVal = X[:,1]
        # ssVal = (X[:, 1] * np.exp((-discRate * X[:, 2])))
        llVal = (X[:, 3] * np.exp((-discRate * X[:, 4])))

        pLL = 1.0 / (1.0 + np.exp(-rho * (llVal - ssVal)))

        return pLL

    def predict_proba(self, X, params=[]):
        return self.choice(X, params)

    def predict(self, X, params=[]):
        return self.choice(X, params).round()

    def set_params(self, discountRate=-5, rho=.01):
        self.discountRate = discountRate
        self.rho = rho
        return self

    def get_params(self, deep=True):
        return {'discountRate': self.discountRate, 'rho': self.rho}


class HyperbolicClassifier(object):
    def __init__(self, discountRate=-5, rho=.01):
        self.discountRate = discountRate
        self.rho = rho

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        ranges = [[-8, .01, .1], [0, 2, .1]]
        opt = scipy.optimize.brute(self.sse, ranges, args=(X, y,))
        self.discountRate = opt[0]
        self.rho = opt[1]
        return self

    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > -.01) or (params[1] < 0.) or (params[1] > 2):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1 - boundary)

        asdf1 = -y * np.log(yhat)
        asdf2 = (1 - y) * np.log(1 - yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        # print([params, sse])
        return sse

    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]

        ssVal = X[:, 1]
        #ssVal = (X[:, 0] * 1 / (1 + discRate * X[:, 1]))
        llVal = (X[:, 3] * (1 / (1 + discRate * X[:, 4])))

        pLL = 1 / (1 + np.exp(-rho * (llVal - ssVal)))

        return pLL

    def predict_proba(self, X, params=[]):
        return self.choice(X, params)

    def predict(self, X, params=[]):
        return self.choice(X, params).round()

    def set_params(self, discountRate=-5, rho=.01):
        self.discountRate = discountRate
        self.rho = rho
        return self

    def get_params(self, deep=True):
        return {'discountRate': self.discountRate, 'rho': self.rho}


class QuasiHyperbolicClassifier(object):
    def __init__(self, discountRate=-5, rho=.01, bias=.01):
        self.discountRate = discountRate
        self.rho = rho
        self.bias = bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        ranges = [[-8, .01, .1], [0, 1.1, .1], [0.1, 1, .1]]
        opt = scipy.optimize.brute(self.sse, ranges, args=(X, y,))
        self.discountRate = opt[0]
        self.rho = opt[1]
        self.bias = opt[2]
        return self

    def sse(self, params, X, y):
        if (params[0] < -7.3) or (params[0] > -.01) or (params[1] <= 0.) or (params[2] > 1) or (
                params[2] < 0):
            # return something 'bad' (i.e., big)
            return 10000000000000
        yhat = self.choice(X, params)

        boundary = .00000001
        yhat = np.clip(yhat, boundary, 1 - boundary)

        asdf1 = -y * np.log(yhat)
        asdf2 = (1 - y) * np.log(1 - yhat)
        sse = np.sum(np.power(asdf1 - asdf2, 2))
        # print([params, sse])
        return sse

    def choice(self, X, params=[]):
        # if params are NOT provided, use internally stored values
        if len(params) == 0:
            rho = self.rho
            discRate = np.exp(self.discountRate)
            bias = self.bias
        # if params ARE provided, use them
        else:
            discRate = np.exp(params[0])
            rho = params[1]
            bias = params[2]

        if bias == 0:
            ssVal = X[:, 1]
            #ssVal = (X[:, 0] * numpy.exp((-discRate * X[:, 1])))
            llVal = (X[:, 3] * np.exp((-discRate * X[:, 4])))
        else:
            ssVal = X[:, 1]
            #ssVal = (X[:, 0] * bias * numpy.exp((-discRate * X[:, 1])))
            llVal = (X[:, 3] * bias * np.exp((-discRate * X[:, 4])))

        pLL = 1 / (1 + np.exp(-rho * (llVal - ssVal)))

        return pLL

    def predict_proba(self, X, params=[]):
        return self.choice(X, params)

    def predict(self, X, params=[]):
        return self.choice(X, params).round()

    def set_params(self, discountRate=-5, rho=.01, bias=.01):
        self.discountRate = discountRate
        self.rho = rho
        self.bias = bias
        return self

    def get_params(self, deep=True):
        return {'discountRate': self.discountRate, 'rho': self.rho, 'bias': self.bias}





st.title('Discount Rate Estimator')

st.markdown("""
Select intertemporal choices from the left side panel and each model will estimate your discounting rate and predict your choices.
""")
numbers = list(range(0,101))
day_df = pd.DataFrame(numbers,columns=['days'])


option = st.selectbox(
    'Select Intertemporal Choice Model',
     ('Exponential', 'Hyperbolic','Quasi-hyperbolic'))

if option == 'Exponential':
    model = ExponentialClassifier()
elif option == 'Hyperbolic':
    model = HyperbolicClassifier()
elif option == 'Quasi-hyperbolic':
    model = QuasiHyperbolicClassifier()

with st.form(key ='Form1'):
    with st.sidebar:

        st.title('Intertemporal Preferences')
        trial_1 = st.radio("Which reward do you prefer?", ('$15 today ','$64 in 40 days'), key = 'trial_1')
        trial_2 = st.radio("Which reward do you prefer?", ('$40 today ','$67 in 55 days'), key = 'trial_2')
        trial_3 = st.radio("Which reward do you prefer?", ('$12 today ','$28 in 60 days'),key = 'trial_3')
        trial_4 = st.radio("Which reward do you prefer?", ('$10 today ','$95 in 35 days'),key = 'trial_4')
        trial_5 = st.radio("Which reward do you prefer?", ('$30 today ','$75 in 92 days'),key = 'trial_5')
        trial_6 = st.radio("Which reward do you prefer?", ('$32 today ','$47 in 75 days'),key = 'trial_6')
        trial_7 = st.radio("Which reward do you prefer?", ('$25 today ','$58 in 55 days'), key = 'trial_7')
        trial_8 = st.radio("Which reward do you prefer?", ('$47 today ','$58 in 80 days'), key = 'trial_8')
        trial_9 = st.radio("Which reward do you prefer?", ('$34 today ','$35 in 73 days'), key = 'trial_9')
        trial_10 = st.radio("Which reward do you prefer?", ('$15 today ','$43 in 44 days'), key = 'trial_10')
        trial_11 = st.radio("Which reward do you prefer?", ('$22 today ','$120 in 38 days'), key = 'trial_11')
        trial_12 = st.radio("Which reward do you prefer?", ('$20 today ','$26 in 45 days'), key = 'trial_12')
        trial_13 = st.radio("Which reward do you prefer?", ('$27 today ','$29 in 65 days'), key = 'trial_13')
        trial_14 = st.radio("Which reward do you prefer?", ('$35 today ','$55 in 70 days'), key = 'trial_14')
        trial_15 = st.radio("Which reward do you prefer?", ('$20 today ','$62 in 55 days'), key = 'trial_15')
        trial_16 = st.radio("Which reward do you prefer?", ('$50 today ','$98 in 100 days'), key = 'trial_16')
        trial_17 = st.radio("Which reward do you prefer?", ('$25 today ','$30 in 65 days'), key = 'trial_17')
        trial_18 = st.radio("Which reward do you prefer?", ('$67 today ','$88 in 65 days'), key = 'trial_18')
        trial_19 = st.radio("Which reward do you prefer?", ('$10 today ','$89 in 37 days'), key = 'trial_19')
        trial_20 = st.radio("Which reward do you prefer?", ('$40 today ','$48 in 58 days'), key = 'trial_20')
        trial_21 = st.radio("Which reward do you prefer?", ('$20 today ','$65 in 78 days'), key = 'trial_21')
        trial_22 = st.radio("Which reward do you prefer?", ('$32 today ','$93 in 50 days'), key = 'trial_22')
        trial_23 = st.radio("Which reward do you prefer?", ('$24 today ','$68 in 45 days'), key = 'trial_23')
        trial_24 = st.radio("Which reward do you prefer?", ('$83 today ','$86 in 65 days'), key = 'trial_24')
        submitted1 = st.form_submit_button(label='Submit Your Choices')

data = {'trial': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        'ssr': [15, 40, 12, 10, 30, 32, 25, 47, 34, 15, 22, 20, 27, 35, 20, 50, 25, 67, 10, 40, 20, 32, 24, 83],
        'ssd': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'llr': [64, 67, 28, 95, 75, 47, 58, 58, 35, 43, 120, 26, 29, 55, 62, 98, 30, 88, 89, 48, 65, 93, 68, 86],
        'lld': [40, 55, 60, 35, 92, 75, 55, 80, 73, 44, 38, 45, 65, 70, 55, 100, 65, 65, 37, 58, 78, 50, 40, 65],
        'selection': [trial_1,trial_2,trial_3,trial_4,trial_5,trial_6,trial_7,trial_8, trial_9,
                   trial_10, trial_11, trial_12, trial_13, trial_14, trial_15, trial_16,
                   trial_17, trial_18, trial_19, trial_20, trial_21, trial_22, trial_23, trial_24]}

data = pd.DataFrame(data=data)
data['choice'] = np.where(data['selection'].str.contains('today'),0,1)
# if calculation == '$25 today':
#     st.write('You selected $25 today')
# else:
#     st.write('you selected $45 in 21 days')
# st.write(calculation)

##here is where we will create the dataframe

#data = [['ssr', 25], ['ssd', 0], ['llr', 45], ['lld', 21]]
def Average(lst):
     return sum(lst) / len(lst)
accuracy_list = []
brier_list = []
disc_list = []

if submitted1:

    #build the X,y split
    X = data.drop(columns = ['selection','choice'])
    X = X.values
    y = data['choice']
    y = y.values



    #set up the kfold split
    kf = KFold(n_splits=10)
    folds = kf.get_n_splits(X)
    count_folds = 0
    total_for_mean_accuracy = 0
    total_for_brier_scores = 0
    total_for_params = 0

    for train_index, test_index in kf.split(X):


        # print('\tStarting new fold ' + str(time.time()) + '\n')

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # create fold count
        count_folds = count_folds + 1
        # print('In a new fold' + '\t'),
        # print(str(count_folds) + '\t'),
        # fit model to training data
        model.fit(X_train, y_train)
        # get the parameters per fold
        # print(str(model.get_params()) + '\t'),
        # score model on test data (using accuracy)
        accuracy_scores = accuracy_score(y_test, model.predict(X_test))
        # print('Accuracy' + '\t'),
        # sys.stdout.write(str(accuracy_scores) +'\t'),
        # calculate the brier scores


        brier_scores = np.power(np.subtract(y_test, model.predict_proba(X_test)), 2)

        mean_brier_scores = np.mean(brier_scores)

        #print(accuracy_scores)

        accuracy_list.append(accuracy_scores)
        brier_list.append(mean_brier_scores)
        #my_list = [elem[0] for elem in model.get_params().values()]

# def list_average(num):
#     sum_num = 0
#     for t in num:
#         sum_num = sum_num + t
#
#     avg = sum_num / len(num)
#     return avg
        values_view = model.get_params().values()
        value_iterator = iter(values_view)
        first_value = next(value_iterator)
        disc_list.append(first_value)

        day_df['disc_rate'] = np.exp(Average(disc_list))
        if option == 'Exponential':
            day_df['present_value'] = 100 * np.exp(day_df['days'] * -day_df['disc_rate'])
        if option == 'Hyperbolic':
            day_df['present_value'] = 100 * (1 / (1 + (day_df['disc_rate'] * day_df['days'])))
        if option == 'Quasi-hyperbolic':
            biased_params = model.get_params().get('bias')
            if biased_params == 0:
                day_df['present_value_100$'] = 100 * np.exp(day_df['days'] * day_df['disc_rate'])
            else:
                day_df['present_value'] = 100 * biased_params * np.exp(day_df['days'] * -day_df['disc_rate'])

        # print(accuracy_list)
    # print(len(accuracy_list))
    st.title("{} Discount Rate: {:.2f} ".format(option, np.exp(Average(disc_list))))
    st.title("Percentage of choices predicted correctly: {:.2f}%".format(Average(accuracy_list) * 100))

    if option == 'Quasi-hyperbolic':
        st.title("Present Bias: {:.2f} ".format(biased_params))
    if option == 'Quasi-hyperbolic':
        st.subheader("Quasi-hyperbolic discounting model:")
        r'''$$present value = Vo *  \delta * e^{-kd} \\$$'''
        st.write('The Quasi-hyperbolic discounting model represents a decision maker who hold dynamically consistent or inconsistent preferences. Its unique feature is the present-bias parameter. As this parameter approaches 0, the decision maker values rewards that are available in present time more disproportionately (dynamically inconsistent), as it approaches 1, the deciison maker values rewards more proportionately (dynamically consistent). If present bias parameter is 1, then the model simply represents the exponential discounting model.')

    if option == 'Hyperbolic':
        st.subheader("Hyperbolic discounting model:")
        r'''$$present value = Vo * \frac{1}{1+kd}\ \\$$'''
        st.write('Unliked the exponential discounting model, the Hyperbolic Discounting Model represents a decision-maker whose preferences can be dynamically inconsistent (but not always), in which rewards become disproportionately more valuable as they become closer to present time.')

    elif option == 'Exponential':
        st.subheader("Exponential discounting model:")
        r'''$$present value = Vo * e^{-kd} \\$$'''
        st.write(
            'First introduced by Paul Samuelson in the 1930s, the exponential discounting model represents a decision-maker who has dynamically consistent intertemporal preferences, indicating that preferences do not change as they move through time, which prevents the decision-maker from being arbitraged.')

    present_value = day_df[['days','present_value']]
    fig, ax = plt.subplots()
    ax = sns.lineplot(data=present_value, x="days", y="present_value")
    ax.set_title("{} Discounted Value of $100".format(option))
    st.pyplot(fig)
    st.table(data)


#st.write(list_average(brier_list))
