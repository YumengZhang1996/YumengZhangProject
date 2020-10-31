import numpy as np
import pandas as pd
import quantopian.optimize as opt
import quantopian.algorithm as algo
import statsmodels.api as sm

 
MAX_GROSS_EXPOSURE = 1.0 # Set exposure constraint constant value for optimizer
    
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    schedule_function(check_pair_status, date_rules.every_day(), time_rules.market_close(minutes=30))
    
  

    context.stocks = [(symbol('MMM'),symbol('DAL')),
(symbol('DDD'),symbol('P')),
(symbol('UPS'),symbol('DAL'))]

    # Our threshold for trading on the z-score
    context.entry_threshold = 2
    context.exit_threshold = 0.05
    
    # Create a variable to store our target weights
    context.target_weights = pd.Series(index=context.stocks, data=0.0)
    
    # Moving average lengths
    context.long_ma_length = 30
    context.short_ma_length = 1
    
    # Flags to tell us if we're currently in a trade
    context.currently_long_the_spread = False
    context.currently_short_the_spread = False


def check_pair_status(context, data):
    
    for pair in context.stocks:
    # For notational convenience
        s1 = pair[0]
        s2 = pair[1]
    # Get pricing history
        prices = data.history([s1, s2], "price", context.long_ma_length, '1d')
        
        #try:
            #hedge_r = hedge_ratio(prices[s1], prices[s2], add_const=True)      
        #except ValueError as e:
            #log.debug(e)
            #return
    

        short_prices = prices.iloc[-context.short_ma_length:]
    
    # Get the long mavg
        long_ma = np.mean(prices[s1] - prices[s2])
                          #hedge_r * prices[s2])
    # Get the std of the long window
        long_std = np.std(prices[s1] - prices[s2])
                          #hedge_r * prices[s2])
        
    
    
    # Get the short mavg
        short_ma = np.mean(short_prices[s1] - short_prices[s2])
                           #hedge_r * short_prices[s2])
    
    # Compute z-score
        if long_std > 0:
            zscore = (short_ma - long_ma)/long_std
    
        # Our two entry cases
            if zscore > context.entry_threshold and \
                not context.currently_short_the_spread:
                context.target_weights[s1] = -1/(2*len(context.stocks)) # short top
                context.target_weights[s2] = 1/(2*len(context.stocks))#*hedge_r # long bottom
                context.currently_short_the_spread = True
                context.currently_long_the_spread = False
                allocate(context, data)
                return
            
            elif zscore < -context.entry_threshold and \
                not context.currently_long_the_spread:
                context.target_weights[s1] = 1/(2*len(context.stocks)) 
                context.target_weights[s2] = -1/(2*len(context.stocks))#*hedge_r
                context.currently_short_the_spread = False
                context.currently_long_the_spread = True
                allocate(context, data)
                return
            
        # Our exit case
            elif abs(zscore) < context.exit_threshold:
                context.target_weights[s1] = 0 # close out
                context.target_weights[s2] = 0 # close out
                context.currently_short_the_spread = False
                context.currently_long_the_spread = False
                allocate(context, data)
                return
                
            record('zscore', zscore)
    
    # Call the tra
    
#def hedge_ratio(Y, X, add_const=True):
    #if add_const:
        #X = sm.add_constant(X)
        #model = sm.OLS(Y, X).fit()
        #return model.params[1]
    #model = sm.OLS(Y, X).fit()
    #return model.params.values        
        
def allocate(context, data):    
    # Set objective to match target weights as closely as possible, given constraints
    objective = opt.TargetWeights(context.target_weights)
    
    # Define constraints
    constraints = []
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_EXPOSURE))
    
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )