data
{
  int<lower=1> n_trainvid;                                     // number of training sequences
  int<lower=1> n_testvid;                                      // number of test sequences

  int<lower=1> n_layers;                                       // model hyper-parameter
  int<lower=1> n_state;                                        // model hyper-parameter

  int num1;                                                    // defined in pystan code
  int num2;                                                    // defined in pystan code
  int num4;                                                    // defined in pystan code

  int<lower=1> n_feature;                                      // input dimension |x_t|

  int<lower=1> n_class;                                        // number of class labels
 
  int<lower=1> n_step[n_trainvid];                             // number of instances in the training sequences
  int<lower=1> test_step_length[n_testvid];                    // number of instances in the test sequences
  int<lower=1> max_step;                                       // maximum number of instances in a sequence

  vector[n_feature] X_train[n_trainvid, max_step];             // training features equivalent of X_train[n_trainvid, max_step, n_feature] in Python

  int<lower=0,upper=4> y[n_trainvid, max_step];                //copy for multilabel problems

  vector[n_feature] X_test[n_testvid, max_step];               // test features equivalent of X_test[n_testvid, max_step, n_feature] in Python

  real reg_l2_transition;                                      // regularization constant for transition and influence parameters
  real reg_l2_feature;                                         // regularization constant for emission/observation parameters
}
 
parameters
{
  simplex[n_label * n_state] theta_transition[n_layers, n_label * n_state];      // transition parameters

  vector[num2] theta_intrinsic;                                                  // influence parameters

  vector[n_feature] theta_feature_cont[n_layers, n_label * n_state];             // emission/observation parameters
}
 
model
{
    vector[num1] alpha_con[max_step];
    vector[num2] alpha_uncon[max_step];

    int zeroth_label = 1;

    int last_label;

    real temp_con[n_layers, n_state, n_state];
    real temp_uncon[n_layers, n_label * n_state, n_label * n_state];   

    int states_cur[n_layers];
    int states_cur_relative[n_layers];
    int states_prev[n_layers];
    int whole_state_cur;
    int temp_state_cur;
    int whole_state_prev;
    int temp_state_prev;

    int actual_state_cur;
    int actual_state_prev;

    int actual_state;
    real temp;

    int num3;
    
    for (i in 1:n_layers)
    {
      for (j in 1:n_state)
      {
        for (k in 1:n_state)
        {
          temp_con[i,j,k] = 0;
        }
      }
    }

    for (i in 1:n_layers)
    {
      for (j in 1:(n_label*n_state))
      {
        for (k in 1:(n_label*n_state))
        {
          temp_uncon[i,j,k] = 0;
        }
      }
    }

    for (i in 1:max_step)
    {
      for (j in 1:num2)
      {
          alpha_uncon[i,j] = 0;
      }
    }
    for (i in 1:max_step)
    {
      for (j in 1:num1)
      {
          alpha_con[i,j] = 0;
      }
    }
    
    for (i in 1:n_trainvid)
    {
        //Compute alpha

        for (j in 1:(n_step[i]))
        {
            int label_previous;
            int label_current = y[i,j];

            if (j == 1)
              label_previous = zeroth_label;
            else
              label_previous = y[i,j-1];

            //constrained
            for (layer in 1:n_layers)
            {
              for (state_cur in 1:n_state)
              {
                actual_state_cur = (label_current - 1) * n_state + state_cur;
                temp = theta_feature_cont[layer, actual_state_cur]' * X_train[i,j];

                for (state_prev in 1:n_state)
                {
                  actual_state_prev = (label_previous - 1) * n_state + state_prev;
                  temp_con[layer, state_cur, state_prev] =  temp + theta_transition[layer, actual_state_prev, actual_state_cur];
                }         
              }
            }

            for (init in 1:num1)
            {
                whole_state_cur = init;
                temp_state_cur = whole_state_cur - 1;
                temp = 0;
                actual_state = 0;
                num3 = 1;

                for (layer in 1:n_layers)
                {
                  states_cur[layer] = (temp_state_cur % n_state) + 1 + (label_current - 1) * n_state;
                  states_cur_relative[layer] = (temp_state_cur % n_state) + 1;
                  temp_state_cur = temp_state_cur / n_state;
                }

                for (layer in 1:n_layers)
                {
                  num3 = num3 * (n_label*n_state);
                  actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label*n_state));
                }

                actual_state = actual_state + 1;

                //temp = temp + theta_intrinsic[actual_state];

                for (layer in 1:n_layers)
                {
                  alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(temp_con[layer, states_cur_relative[layer]]);
                }
                alpha_con[j,init] = alpha_con[j,init] + theta_intrinsic[actual_state];

                if (j > 1)
                  alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(alpha_con[j-1]);
            }

            //unconstrained
            for (layer in 1:n_layers)
            {
              for (state_cur in 1:(n_label*n_state))
              {
                actual_state_cur = state_cur;
                temp = theta_feature_cont[layer, actual_state_cur]' * X_train[i,j];

                for (state_prev in 1:(n_label*n_state))
                {
                  actual_state_prev = state_prev;
                  temp_uncon[layer, state_cur, state_prev] =  temp + theta_transition[layer, actual_state_prev, actual_state_cur];
                }         
              }
            }

            for (init in 1:num2)
            {
                whole_state_cur = init;
                temp_state_cur = whole_state_cur - 1;
                temp = 0;
                actual_state = 0;
                num3 = 1;

                for (layer in 1:n_layers)
                {
                  states_cur[layer] = (temp_state_cur % (n_label*n_state)) + 1;
                  temp_state_cur = temp_state_cur / (n_label*n_state);
                }

                for (layer in 1:n_layers)
                {
                  num3 = num3 * (n_label*n_state);
                  actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label*n_state));
                }

                actual_state = actual_state + 1;

                //temp = temp + theta_intrinsic[actual_state];

                for (layer in 1:n_layers)
                {
                  alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon[layer, states_cur[layer]]);
                }
                alpha_uncon[j,init] = alpha_uncon[j,init] + theta_intrinsic[actual_state];

                if (j > 1)
                  alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(alpha_uncon[j-1]);
            }

            // reset temp_con and temp_uncon

            for (i_layer in 1:n_layers)
            {
              for (j_state in 1:n_state)
              {
                for (k in 1:n_state)
                {
                  temp_con[i_layer,j_state,k] = 0;
                }
              }
            }

            for (i_layer in 1:n_layers)
            {
              for (j_state in 1:(n_label*n_state))
              {
                for (k in 1:(n_label*n_state))
                {
                  temp_uncon[i_layer,j_state,k] = 0;
                }
              }
            }
        }

        target += log_sum_exp(alpha_con[n_step[i]]) - log_sum_exp(alpha_uncon[n_step[i]]);

        for (j in 1:max_step)
        {
          for (k in 1:num2)
          {
              alpha_uncon[j,k] = 0;
          }
        }
        for (j in 1:max_step)
        {
          for (k in 1:num1)
          {
              alpha_con[j,k] = 0;
          }
        }
    } 

    //add regularisation

    for (layer in 1:n_layers)
    {
      for (i_state in 1:(n_label * n_state))
      {
        for (feature in 1:n_feature)
        {
          target += -reg_l2_feature * theta_feature_cont[layer, i_state, feature] ^ 2;
        }
        for (i_prior_state in 1:(n_label * n_state))
        {
           target += -reg_l2_transition * theta_transition[layer, i_prior_state, i_state] ^ 2;
        }
      }
    }

    for (state in 1:num2)
    {
      target += -reg_l2_transition * theta_intrinsic[state] ^ 2;
    }
     
}

generated quantities
{
  //using online inference(forward algorithm)

vector[n_label] Prob_y[n_testvid,max_step];
vector[n_label] transformed_proby[n_testvid,max_step];
vector[n_label] Actual_proby[n_testvid,max_step];

int label_previous = 1;
real max_prob;

vector[num2] alpha_uncon[max_step];

real temp_uncon[n_layers, n_label * n_state, n_label * n_state]; 
vector[n_state*num4] temp_alpha_store;

int states_cur[n_layers];
int states_prev[n_layers];
int whole_state_cur;
int temp_state_cur;
int whole_state_prev;
int temp_state_prev;

int actual_state_cur;
int actual_state_prev;

int actual_state;
real temp;

int num3;

for (i_layer in 1:n_layers)
{
  for (j_state in 1:(n_label*n_state))
  {
    for (k in 1:(n_label*n_state))
    {
      temp_uncon[i_layer,j_state,k] = 0;
    }
  }
}

for(i in 1:n_testvid)
{
  for (j in 1:max_step)
  {
    for (l in 1:n_label)
    {
      Prob_y[i,j,l] = 0;
    }
  }
}

// start predicting using trained model

for(i in 1:n_testvid)
{
  // print("video", i);
   
  for (j in 1:max_step)
  {
    for (k in 1:num2)
    {
      alpha_uncon[j,k] = 0;
    }
  }
  
  for (j in 1:test_step_length[i])
  { 
    for (layer in 1:n_layers)
    {
      for (state_cur in 1:(n_label*n_state))
      {
        actual_state_cur = state_cur;
        temp = theta_feature_cont[layer, actual_state_cur]' * X_test[i,j];

        for (state_prev in 1:(n_label*n_state))
        {
          actual_state_prev =  state_prev;
          temp_uncon[layer, state_cur, state_prev] =  temp + theta_transition[layer, actual_state_prev, actual_state_cur];
        }         
      }
    }

    for (init in 1:num2)
    {
      whole_state_cur = init;
      temp_state_cur = whole_state_cur - 1;
      temp = 0;
      actual_state = 0;
      num3 = 1;

      for (layer in 1:n_layers)
      {
        states_cur[layer] = (temp_state_cur % (n_label*n_state)) + 1;
        temp_state_cur = temp_state_cur / (n_label*n_state);
      }

      for (layer in 1:n_layers)
      {
        num3 = num3 * (n_label*n_state);
        actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label*n_state));
      }

      actual_state = actual_state + 1;

      //temp = temp + theta_intrinsic[actual_state];

      for (layer in 1:n_layers)
      {
        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon[layer, states_cur[layer]]);
      }
      alpha_uncon[j,init] = alpha_uncon[j,init] + theta_intrinsic[actual_state];

      if (j > 1)
        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(alpha_uncon[j-1]);
    }   

    //inference
    for (label in 1:n_label)
    {
      temp_alpha_store[1:(n_state*num4)] = alpha_uncon[j,(n_state*(label-1)*num4 + 1):(n_state*label*num4)];    //alpha[j,(label-1)*3+store];

      Prob_y[i,j,label] = log_sum_exp(temp_alpha_store);

      for (store in 1:(n_state*num4))
      {
        temp_alpha_store[store] = 0;
      }   
    }

    for (i_layer in 1:n_layers)
    {
      for (j_state in 1:(n_label*n_state))
      {
        for (k in 1:(n_label*n_state))
        {
          temp_uncon[i_layer,j_state,k] = 0;
        }
      }
    }
  }
  for (j in 1:test_step_length[i])
  {
    max_prob = max(Prob_y[i, j]);
    for (label in 1:n_label)
    {
      transformed_proby[i,j,label] = exp(Prob_y[i, j, label] - max_prob);
    }
    for (label in 1:n_label)
    {
      Actual_proby[i,j,label] = transformed_proby[i,j,label]/(sum(transformed_proby[i,j]));  
    }
  }
}

}