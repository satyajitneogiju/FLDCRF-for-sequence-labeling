data
{
  int<lower=1> n_trainvid;       // number of training sequences

  int<lower=1> n_testvid;        // number of test sequences

  int<lower=1> n_layers;         // number of hidden layers, in this code equal to number of different label categpries

  //int<lower=1> n_label_types;   // n_object_types for social interaction (use in interaction model, uncomment)

  int num1;                       // defined in python code
  int num2;                       // defined in python code
  // int num4;

  int<lower=1> n_feature_cont;    //input dimension |x_t|

  int<lower=1> n_state[n_layers];  // number of hidden states per class label in each layer
  int<lower=1> n_label[n_layers];  // number of classes in each label category
 
  int<lower=1> n_step[n_trainvid];   // number of instances in the training sequences

  int<lower=1> test_step_length[n_testvid];    // number of instances in the test sequences
  int<lower=1> max_step;                       // maximum number of instances in a sequence

  vector[n_feature_cont] X_train_cont[n_trainvid, max_step];        // training features equivalent of X_train[n_trainvid, max_step, n_feature] in Python
   
  int<lower=0,upper=4> y[n_trainvid, max_step, n_layers];           // training labels 

  //int<lower=0,upper=10> label_type[n_trainvid];                   

  vector[n_feature_cont] X_test_cont[n_testvid, max_step];          // test features equivalent of X_test[n_testvid, max_step, n_feature] in Python

  real reg_l2_transition;                                           // regularization constant for transition and influence parameters
  real reg_l2_feature;                                              // regularization constant for emission/observation parameters
}
 
parameters
{
  simplex[n_label[1] * n_state[1]] theta_transition1[n_label[1] * n_state[1]];         // transition parameters layer 1
  simplex[n_label[2] * n_state[2]] theta_transition2[n_label[2] * n_state[2]];         // transition parameters layer 2

  vector[n_label[2] * n_state[2]] theta_transition12[n_label[1] * n_state[1]];        // influence parameters cross link 1
  vector[n_label[1] * n_state[1]] theta_transition21[n_label[2] * n_state[2]];        // influence parameters cross link 2

  vector[num2] theta_intrinsic;                                                        // influence parameters cotemporal

  vector[n_feature_cont] theta_feature_cont1[n_label[1] * n_state[1]];                 // emission/observation parameters layer 1
  vector[n_feature_cont] theta_feature_cont2[n_label[2] * n_state[2]];                 // emission/observation parameters layer 2 (extend for 3 or more label categories)
}
 
model
{
    vector[num1] alpha_con[max_step];
    vector[num2] alpha_uncon[max_step];

    int zeroth_label = 1;

    int last_label;

    real temp_con1[n_state[1], n_state[1]];
    real temp_con2[n_state[2], n_state[2]];

    real temp_con12[n_state[2], n_state[1]];   // for layer 1 (t-1) to layer 2 (t), likelihood cliques computed in layer 2
    real temp_con21[n_state[1], n_state[2]];   // for layer 2 (t-1) to layer 1 (t), likelihood cliques computed in layer 1

    real temp_uncon1[n_label[1] * n_state[1], n_label[1] * n_state[1]]; 
    real temp_uncon2[n_label[2] * n_state[2], n_label[2] * n_state[2]];   

    real temp_uncon12[n_label[2] * n_state[2], n_label[1] * n_state[1]]; 
    real temp_uncon21[n_label[1] * n_state[1], n_label[2] * n_state[2]]; 

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
    real temp1;
    real temp2;

    int num3;
    
    for (j in 1:n_state[1])
    {
      for (k in 1:n_state[1])
      {
        temp_con1[j,k] = 0;
      }
    }

    for (j in 1:n_state[2])
    {
      for (k in 1:n_state[2])
      {
        temp_con2[j,k] = 0;
      }
    }

    for (j in 1:n_state[2])
    {
      for (k in 1:n_state[1])
      {
        temp_con12[j,k] = 0;
      }
    }

    for (j in 1:n_state[1])
    {
      for (k in 1:n_state[2])
      {
        temp_con21[j,k] = 0;
      }
    }

    for (j in 1:(n_label[1]*n_state[1]))
    {
      for (k in 1:(n_label[1]*n_state[1]))
      {
        temp_uncon1[j,k] = 0;
      }
    }

    for (j in 1:(n_label[2]*n_state[2]))
    {
      for (k in 1:(n_label[2]*n_state[2]))
      {
        temp_uncon2[j,k] = 0;
      }
    }

    for (j in 1:(n_label[2]*n_state[2]))
    {
      for (k in 1:(n_label[1]*n_state[1]))
      {
        temp_uncon12[j,k] = 0;
      }
    }

    for (j in 1:(n_label[1]*n_state[1]))
    {
      for (k in 1:(n_label[2]*n_state[2]))
      {
        temp_uncon21[j,k] = 0;
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
            int label_previous[2];
            int label_current[2];

            label_current[1] = y[i,j,1];
            label_current[2] = y[i,j,2];

            if (j == 1)
            {
              label_previous[1] = zeroth_label;
              label_previous[2] = zeroth_label;
            }
            else
            {
              label_previous[1] = y[i,j-1,1];
              label_previous[2] = y[i,j-1,2];
            }

            //constrained
            for (state_cur in 1:n_state[1])
            {
              actual_state_cur = (label_current[1] - 1) * n_state[1] + state_cur;
              temp1 = theta_feature_cont1[actual_state_cur]' * X_train_cont[i,j];

              for (state_prev in 1:n_state[1])
              {
                actual_state_prev = (label_previous[1] - 1) * n_state[1] + state_prev;
                temp_con1[state_cur, state_prev] =  temp1 + theta_transition1[actual_state_prev, actual_state_cur];
              }

              for (state_prev in 1:n_state[2])
              {
                actual_state_prev = (label_previous[2] - 1) * n_state[2] + state_prev;
                temp_con21[state_cur, state_prev] =  theta_transition21[actual_state_prev, actual_state_cur];
              }
            }

            for (state_cur in 1:n_state[2])
            {
              actual_state_cur = (label_current[2] - 1) * n_state[2] + state_cur;
              temp2 = theta_feature_cont2[actual_state_cur]' * X_train_cont[i,j];

              for (state_prev in 1:n_state[2])
              {
                actual_state_prev = (label_previous[2] - 1) * n_state[2] + state_prev;
                temp_con2[state_cur, state_prev] =  temp2 + theta_transition2[actual_state_prev, actual_state_cur];
              }

              for (state_prev in 1:n_state[1])
              {
                actual_state_prev = (label_previous[1] - 1) * n_state[1] + state_prev;
                temp_con12[state_cur, state_prev] =  theta_transition12[actual_state_prev, actual_state_cur];
              }
            }

            for (init in 1:num1)
            {
                whole_state_cur = init;
                temp_state_cur = whole_state_cur - 1;
                actual_state = 0;
                num3 = 1;

                for (layer in 1:n_layers)
                {
                  states_cur[layer] = (temp_state_cur % n_state[layer]) + 1 + (label_current[layer] - 1) * n_state[layer];
                  states_cur_relative[layer] = (temp_state_cur % n_state[layer]) + 1;
                  temp_state_cur = temp_state_cur / n_state[layer];
                }

                for (layer in 1:n_layers)
                {
                  num3 = num3 * (n_label[layer]*n_state[layer]);
                  actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label[layer]*n_state[layer]));
                }

                actual_state = actual_state + 1;

                //temp = temp + theta_intrinsic[actual_state];

                alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(temp_con1[states_cur_relative[1]]);
                alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(temp_con2[states_cur_relative[2]]);

                alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(temp_con21[states_cur_relative[1]]);
                alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(temp_con12[states_cur_relative[2]]);

                alpha_con[j,init] = alpha_con[j,init] + theta_intrinsic[actual_state];

                if (j > 1)
                  alpha_con[j,init] = alpha_con[j,init] + log_sum_exp(alpha_con[j-1]);
            }


            //unconstrained
            for (state_cur in 1:(n_label[1]*n_state[1]))
            {
              actual_state_cur = state_cur;
              temp1 = theta_feature_cont1[actual_state_cur]' * X_train_cont[i,j];

              for (state_prev in 1:(n_label[1]*n_state[1]))
              {
                actual_state_prev = state_prev;
                temp_uncon1[state_cur, state_prev] =  temp1 + theta_transition1[actual_state_prev, actual_state_cur];
              }

              for (state_prev in 1:(n_label[2]*n_state[2]))
              {
                actual_state_prev = state_prev;
                temp_uncon21[state_cur, state_prev] =  theta_transition21[actual_state_prev, actual_state_cur];
              }
            }

            for (state_cur in 1:(n_label[2]*n_state[2]))
            {
              actual_state_cur = state_cur;
              temp2 = theta_feature_cont2[actual_state_cur]' * X_train_cont[i,j];

              for (state_prev in 1:(n_label[2]*n_state[2]))
              {
                actual_state_prev = state_prev;
                temp_uncon2[state_cur, state_prev] =  temp2 + theta_transition2[actual_state_prev, actual_state_cur];
              }

              for (state_prev in 1:(n_label[1]*n_state[1]))
              {
                actual_state_prev = state_prev;
                temp_uncon12[state_cur, state_prev] =  theta_transition12[actual_state_prev, actual_state_cur];
              }
            }

            for (init in 1:num2)
            {
                whole_state_cur = init;
                temp_state_cur = whole_state_cur - 1;
                //temp = 0;
                actual_state = 0;
                num3 = 1;

                for (layer in 1:n_layers)
                {
                  states_cur[layer] = (temp_state_cur % (n_label[layer]*n_state[layer])) + 1;
                  temp_state_cur = temp_state_cur / (n_label[layer]*n_state[layer]);
                }

                for (layer in 1:n_layers)
                {
                  num3 = num3 * (n_label[layer]*n_state[layer]);
                  actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label[layer]*n_state[layer]));
                }

                actual_state = actual_state + 1;

                //temp = temp + theta_intrinsic[actual_state];

                alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon1[states_cur[1]]);
                alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon2[states_cur[2]]);

                alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon21[states_cur[1]]);
                alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon12[states_cur[2]]);

                alpha_uncon[j,init] = alpha_uncon[j,init] + theta_intrinsic[actual_state];

                if (j > 1)
                  alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(alpha_uncon[j-1]);
            }

            // reset temp_con and temp_uncon

            for (j_state in 1:n_state[1])
            {
              for (k in 1:n_state[1])
              {
                temp_con1[j_state,k] = 0;
              }
            }
            for (j_state in 1:n_state[2])
            {
              for (k in 1:n_state[2])
              {
                temp_con2[j_state,k] = 0;
              }
            }
            for (j_state in 1:n_state[2])
            {
              for (k in 1:n_state[1])
              {
                temp_con12[j_state,k] = 0;
              }
            }
            for (j_state in 1:n_state[1])
            {
              for (k in 1:n_state[2])
              {
                temp_con21[j_state,k] = 0;
              }
            }

            for (j_state in 1:(n_label[1]*n_state[1]))
            {
              for (k in 1:(n_label[1]*n_state[1]))
              {
                temp_uncon1[j_state,k] = 0;
              }
            }
            for (j_state in 1:(n_label[2]*n_state[2]))
            {
              for (k in 1:(n_label[2]*n_state[2]))
              {
                temp_uncon2[j_state,k] = 0;
              }
            }
            for (j_state in 1:(n_label[2]*n_state[2]))
            {
              for (k in 1:(n_label[1]*n_state[1]))
              {
                temp_uncon12[j_state,k] = 0;
              }
            }
            for (j_state in 1:(n_label[1]*n_state[1]))
            {
              for (k in 1:(n_label[2]*n_state[2]))
              {
                temp_uncon21[j_state,k] = 0;
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

      for (i_state in 1:(n_label[1] * n_state[1]))
      {
        for (feature in 1:(n_feature_cont))
        {
          target += -reg_l2_feature * theta_feature_cont1[i_state, feature] ^ 2;
        }
        for (i_prior_state in 1:(n_label[1] * n_state[1]))
        {
           target += -reg_l2_transition * theta_transition1[i_prior_state, i_state] ^ 2;
        }
        for (i_prior_state in 1:(n_label[2] * n_state[2]))
        {
           target += -reg_l2_transition * theta_transition21[i_prior_state, i_state] ^ 2;
        }
      }

      for (i_state in 1:(n_label[2] * n_state[2]))
      {
        for (feature in 1:(n_feature_cont))
        {
          target += -reg_l2_feature * theta_feature_cont2[i_state, feature] ^ 2;
        }
        for (i_prior_state in 1:(n_label[2] * n_state[2]))
        {
           target += -reg_l2_transition * theta_transition2[i_prior_state, i_state] ^ 2;
        }
        for (i_prior_state in 1:(n_label[1] * n_state[1]))
        {
           target += -reg_l2_transition * theta_transition12[i_prior_state, i_state] ^ 2;
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

vector[n_label[1]] Prob_y1[n_testvid,max_step];
vector[n_label[1]] transformed_proby1[n_testvid,max_step];
vector[n_label[1]] Actual_proby1[n_testvid,max_step];

vector[n_label[2]] Prob_y2[n_testvid,max_step];
vector[n_label[2]] transformed_proby2[n_testvid,max_step];
vector[n_label[2]] Actual_proby2[n_testvid,max_step];

int label_previous = 1;
real max_prob1;
real max_prob2;

int count1[n_label[1]];
int count2[n_label[2]];

vector[num2] alpha_uncon[max_step];

real temp_uncon1[n_label[1] * n_state[1], n_label[1] * n_state[1]]; 
real temp_uncon2[n_label[2] * n_state[2], n_label[2] * n_state[2]]; 

real temp_uncon12[n_label[2] * n_state[2], n_label[1] * n_state[1]]; 
real temp_uncon21[n_label[1] * n_state[1], n_label[2] * n_state[2]]; 

vector[n_state[1] * (n_label[2] * n_state[2])] temp_alpha_store1[n_label[1]];
vector[n_state[2] * (n_label[1] * n_state[1])] temp_alpha_store2[n_label[2]];

int states_cur[n_layers];
int states_prev[n_layers];
int whole_state_cur;
int temp_state_cur;
int whole_state_prev;
int temp_state_prev;

int actual_state_cur;
int actual_state_prev;

int actual_state;
real temp1;
real temp2;
int cur_label;

int num3;

for (j_state in 1:(n_label[1]*n_state[1]))
{
  for (k in 1:(n_label[1]*n_state[1]))
  {
    temp_uncon1[j_state,k] = 0;
  }
}
for (j_state in 1:(n_label[2]*n_state[2]))
{
  for (k in 1:(n_label[2]*n_state[2]))
  {
    temp_uncon2[j_state,k] = 0;
  }
}

for (j in 1:(n_label[2]*n_state[2]))
{
  for (k in 1:(n_label[1]*n_state[1]))
  {
    temp_uncon12[j,k] = 0;
  }
}

for (j in 1:(n_label[1]*n_state[1]))
{
  for (k in 1:(n_label[2]*n_state[2]))
  {
    temp_uncon21[j,k] = 0;
  }
}

for(i in 1:n_testvid)
{
  for (j in 1:max_step)
  {
    for (l in 1:n_label[1])
    {
      Prob_y1[i,j,l] = 0;
    }
    for (l in 1:n_label[2])
    {
      Prob_y2[i,j,l] = 0;
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

    for (state_cur in 1:(n_label[1]*n_state[1]))
    {
      actual_state_cur = state_cur;
      temp1 = theta_feature_cont1[actual_state_cur]' * X_test_cont[i,j];

      for (state_prev in 1:(n_label[1]*n_state[1]))
      {
        actual_state_prev = state_prev;
        temp_uncon1[state_cur, state_prev] =  temp1 + theta_transition1[actual_state_prev, actual_state_cur];
      }

      for (state_prev in 1:(n_label[2]*n_state[2]))
      {
        actual_state_prev = state_prev;
        temp_uncon21[state_cur, state_prev] =  theta_transition21[actual_state_prev, actual_state_cur];
      }
    }

    for (state_cur in 1:(n_label[2]*n_state[2]))
    {
      actual_state_cur = state_cur;
      temp2 = theta_feature_cont2[actual_state_cur]' * X_test_cont[i,j];

      for (state_prev in 1:(n_label[2]*n_state[2]))
      {
        actual_state_prev = state_prev;
        temp_uncon2[state_cur, state_prev] =  temp2 + theta_transition2[actual_state_prev, actual_state_cur];
      }

      for (state_prev in 1:(n_label[1]*n_state[1]))
      {
        actual_state_prev = state_prev;
        temp_uncon12[state_cur, state_prev] =  theta_transition12[actual_state_prev, actual_state_cur];
      }
    }

    for (label in 1:n_label[1])
    {
      count1[label] = 0;
    }
    for (label in 1:n_label[2])
    {
      count2[label] = 0;
    }


    for (init in 1:num2)
    {
        whole_state_cur = init;
        temp_state_cur = whole_state_cur - 1;
        //temp = 0;
        actual_state = 0;
        num3 = 1;

        for (layer in 1:n_layers)
        {
          states_cur[layer] = (temp_state_cur % (n_label[layer]*n_state[layer])) + 1;
          temp_state_cur = temp_state_cur / (n_label[layer]*n_state[layer]);
        }

        for (layer in 1:n_layers)
        {
          num3 = num3 * (n_label[layer]*n_state[layer]);
          actual_state = actual_state + (states_cur[layer] - 1) * (num3 / (n_label[layer]*n_state[layer]));
        }

        actual_state = actual_state + 1;

        //temp = temp + theta_intrinsic[actual_state];

        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon1[states_cur[1]]);
        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon2[states_cur[2]]);

        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon21[states_cur[1]]);
        alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(temp_uncon12[states_cur[2]]);

        alpha_uncon[j,init] = alpha_uncon[j,init] + theta_intrinsic[actual_state];

        if (j > 1)
          alpha_uncon[j,init] = alpha_uncon[j,init] + log_sum_exp(alpha_uncon[j-1]);

        for (layer in 1:n_layers)
        {
          cur_label = ((states_cur[layer]-1) / n_state[layer]) + 1;

          if (layer == 1)
          {  
            count1[cur_label] = count1[cur_label] + 1;
            temp_alpha_store1[cur_label, count1[cur_label]] = alpha_uncon[j,init]; 
          }

          if (layer == 2)
          {
            count2[cur_label] = count2[cur_label] + 1;
            temp_alpha_store2[cur_label, count2[cur_label]] = alpha_uncon[j,init];
          }
        }
    }
    

    //inference

    for (layer in 1:n_layers)
    {
      if (layer == 1)  
      {  
        for (label in 1:n_label[1])
        {
          Prob_y1[i,j, label] = log_sum_exp(temp_alpha_store1[label]);

          for (store in 1:(n_state[1]*(n_label[2] * n_state[2])))
          {
            temp_alpha_store1[label, store] = 0;
          }   
        }
      }

      if (layer == 2)  
      {  
        for (label in 1:n_label[2])
        {
          Prob_y2[i,j, label] = log_sum_exp(temp_alpha_store2[label]);

          for (store in 1:(n_state[2]*(n_label[1] * n_state[1])))
          {
            temp_alpha_store2[label, store] = 0;
          }   
        }
      }
    }


    //reset temp_uncon

    for (j_state in 1:(n_label[1]*n_state[1]))
    {
      for (k in 1:(n_label[1]*n_state[1]))
      {
        temp_uncon1[j_state,k] = 0;
      }
    }
    for (j_state in 1:(n_label[2]*n_state[2]))
    {
      for (k in 1:(n_label[2]*n_state[2]))
      {
        temp_uncon2[j_state,k] = 0;
      }
    }  

    for (j_state in 1:(n_label[2]*n_state[2]))
    {
      for (k in 1:(n_label[1]*n_state[1]))
      {
        temp_uncon12[j_state,k] = 0;
      }
    }
    for (j_state in 1:(n_label[1]*n_state[1]))
    {
      for (k in 1:(n_label[2]*n_state[2]))
      {
        temp_uncon21[j_state,k] = 0;
      }
    }

  }
  for (j in 1:test_step_length[i])
  {
    max_prob1 = max(Prob_y1[i, j]);
    max_prob2 = max(Prob_y2[i, j]);

    for (label in 1:n_label[1])
    {
      transformed_proby1[i,j,label] = exp(Prob_y1[i, j, label] - max_prob1);
    }
    for (label in 1:n_label[2])
    {
      transformed_proby2[i,j,label] = exp(Prob_y2[i, j, label] - max_prob2);
    }

    for (label in 1:n_label[1])
    {
      Actual_proby1[i,j,label] = transformed_proby1[i,j,label]/(sum(transformed_proby1[i,j]));  
    }
    for (label in 1:n_label[2])
    {
      Actual_proby2[i,j,label] = transformed_proby2[i,j,label]/(sum(transformed_proby2[i,j]));  
    }

    //print("time ", j, "::", "Layer 1 locomotion:", "stand: ", Actual_proby1[i,j,1], ":walk: ", Actual_proby1[i,j,2], ":sit: ", Actual_proby1[i,j,3], ":lie: ", Actual_proby1[i,j,4]);
    //print("time ", j, "::", "Layer 2 HL:", "Early morning: ", Actual_proby2[i,j,1], ":Relaxing: ", Actual_proby2[i,j,2]);
  }
}

}