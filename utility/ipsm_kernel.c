// Author: Guiming Zhang
// Last update: August 8 2019

#define N_SAMPLES 100

__kernel void iPSM_Predict(const unsigned int nrows_evs, const unsigned int nrows_X, const unsigned int ncols_X, const unsigned int nrows_samples,
                           const unsigned int mode, const float threshold, __global int* MSR_LEVELS, __global float* samples_SD_evs, __global float* SD_evs,
                           __global float* X, __global float* sample_X, __global float* sample_weights, __global float* sample_attributes,
                           __global float* X_predictions, __global float* X_uncertainties)
{
    // this is the ith row (location) in X
    unsigned int i = get_global_id(0);

    // compute the similarity btw ith location to each of the samples
    float similarities[N_SAMPLES];
    for(unsigned int j = 0; j < nrows_samples; j++){ // jth sample
        //if(i == 0)
        //  printf("sample_X_%d = ", j);
        // on each covariates
        float min_sim = 99999.0f;
        for(unsigned int k = 0; k < ncols_X; k++){ // kth variable
            float tmp_sim = 0.0f;
            int msrlevel = MSR_LEVELS[k];
            float X_ik = X[i * ncols_X + k];
            float sample_X_jk = sample_X[j * ncols_X + k];

            //if(i == 0)
            //  printf("%f ", sample_X_jk);

            if(msrlevel == 0 || msrlevel == 1){
                if(X_ik == sample_X_jk)
                    tmp_sim = 1.0f;
                else
                    tmp_sim = 0.0f;
            }
            else{
                float SD_k = SD_evs[k];
                float SD_kj = samples_SD_evs[j * ncols_X + k];
                //if(i == 0)
                //  printf("SD_evj = %f ", SD_kj);
                float diff = (X_ik - sample_X_jk);
                float denom = SD_k * SD_k / SD_kj;
                tmp_sim = exp(-0.5f *  (diff * diff) / (denom * denom));
            }
            //if(i == 0)
            //  printf("%f ", tmp_sim);
            if(tmp_sim < min_sim){
                min_sim = tmp_sim;
            }
        }
        //if(i == 0)
        //  printf("\n");
        similarities[j] = min_sim;
    }

    // now do prediction
    if(mode == 1){ // predict class
        int max_sim_idx = -1;
        float max_sim = 0.0f;
        float max_w = 0.0f;

        // to avoid NoData prediction
        float tmp_threshold = threshold;
        //while(max_sim_idx == -1){

        for(unsigned int j = 0; j < nrows_samples; j++){
            if(similarities[j] >= 1.0f - tmp_threshold){ //apply threshold
                if(sample_weights[j] > max_w){
                    max_w = sample_weights[j];
                }
                float sim = similarities[j] * sample_weights[j];
                if(sim > max_sim){
                    max_sim = sim;
                    max_sim_idx = j;
                }
            }
        }

        X_predictions[i] = sample_attributes[max_sim_idx];
        X_uncertainties[i] = 1.0f - max_sim / max_w;
    }
    else{ // predict property
        float sum_sim = 0.0f;
        float max_sim = 0.0f;
        float sum_weighted_property = 0.0f;
        float max_w = 0.0f;

        // to avoid NoData prediction
        float tmp_threshold = threshold;
        //while(sum_sim == 0.0){

        for(unsigned int j = 0; j < nrows_samples; j++){
          if(similarities[j] >= 1.0f - tmp_threshold){
              if(sample_weights[j] > max_w){
                  max_w = sample_weights[j];
              }
              float sim = similarities[j] * sample_weights[j];
              if(sim > max_sim){
                  max_sim = sim;
              }

              sum_sim += sim;
              sum_weighted_property += sim * sample_attributes[j];

          }
        }

        X_predictions[i] = sum_weighted_property / sum_sim;
        X_uncertainties[i] = 1.0f - max_sim / max_w;

    }

}

__kernel void iPSM_Predict_Sequential(const unsigned int nrows_evs, const unsigned int nrows_X, const unsigned int ncols_X, const unsigned int nrows_samples,
                           const unsigned int mode, const float threshold, __global int* MSR_LEVELS, __global float* samples_SD_evs, __global float* SD_evs,
                           __global float* X, __global float* sample_X, __global float* sample_weights, __global float* sample_attributes,
                           __global float* X_predictions, __global float* X_uncertainties)
{
    for(unsigned int i = 0; i < nrows_X; i++){

      // compute the similarity btw ith location to each of the samples
      float similarities[N_SAMPLES];
      for(unsigned int j = 0; j < nrows_samples; j++){ // jth sample
          //if(i == 0)
          //  printf("sample_X_%d = ", j);
          // on each covariates
          float min_sim = 99999.0f;
          for(unsigned int k = 0; k < ncols_X; k++){ // kth variable
              float tmp_sim = 0.0f;
              int msrlevel = MSR_LEVELS[k];
              float X_ik = X[i * ncols_X + k];
              float sample_X_jk = sample_X[j * ncols_X + k];

              //if(i == 0)
              //  printf("%f ", sample_X_jk);

              if(msrlevel == 0 || msrlevel == 1){
                  if(X_ik == sample_X_jk)
                      tmp_sim = 1.0f;
                  else
                      tmp_sim = 0.0f;
              }
              else{
                  float SD_k = SD_evs[k];

                  float SD_kj = samples_SD_evs[j * ncols_X + k];
                  //if(i == 0)
                  //  printf("SD_evj = %f ", SD_kj);
                  float diff = (X_ik - sample_X_jk);
                  float denom = SD_k * SD_k / SD_kj;
                  tmp_sim = exp(-0.5f *  (diff * diff) / (denom * denom));
              }
              //if(i == 0)
              //  printf("%f ", tmp_sim);
              if(tmp_sim < min_sim){
                  min_sim = tmp_sim;
              }
          }
          //if(i == 0)
          //  printf("\n");
          similarities[j] = min_sim;
      }

      // now do prediction
      if(mode == 1){ // predict class
          int max_sim_idx = -1;
          float max_sim = 0.0f;
          float max_w = 0.0f;

          // to avoid NoData prediction
          float tmp_threshold = threshold;
          //while(max_sim_idx == -1){

          for(unsigned int j = 0; j < nrows_samples; j++){
              if(similarities[j] >= 1.0f - tmp_threshold){ //apply threshold
                  if(sample_weights[j] > max_w){
                      max_w = sample_weights[j];
                  }
                  float sim = similarities[j] * sample_weights[j];
                  if(sim > max_sim){
                      max_sim = sim;
                      max_sim_idx = j;
                  }
              }
          }

          X_predictions[i] = sample_attributes[max_sim_idx];
          X_uncertainties[i] = 1.0f - max_sim / max_w;
      }
      else{ // predict property
          float sum_sim = 0.0f;
          float max_sim = 0.0f;
          float sum_weighted_property = 0.0f;
          float max_w = 0.0f;

          // to avoid NoData prediction
          float tmp_threshold = threshold;
          //while(sum_sim == 0.0){

          for(unsigned int j = 0; j < nrows_samples; j++){
            if(similarities[j] >= 1.0f - tmp_threshold){
                if(sample_weights[j] > max_w){
                    max_w = sample_weights[j];
                }
                float sim = similarities[j] * sample_weights[j];
                if(sim > max_sim){
                    max_sim = sim;
                }
                //if(sim >= 1.0f - threshold){
                sum_sim += sim;
                sum_weighted_property += sim * sample_attributes[j];
                //}
            }
          }

          X_predictions[i] = sum_weighted_property / sum_sim;
          X_uncertainties[i] = 1.0f - max_sim / max_w;

      }
    }
}

__kernel void iPSM_Predict_naive(const unsigned int nrows_evs, const unsigned int nrows_X, const unsigned int ncols_X, const unsigned int nrows_samples,
                           const unsigned int mode, const float threshold, __global int* MSR_LEVELS, __global float* evs, __global float* SD_evs,
                           __global float* X, __global float* sample_X, __global float* sample_weights, __global float* sample_attributes,
                           __global float* X_predictions, __global float* X_uncertainties)
{
    // this is the ith row (location) in X
    unsigned int i = get_global_id(0);

    // compute the similarity btw ith location to each of the samples
    float similarities[N_SAMPLES];
    for(unsigned int j = 0; j < nrows_samples; j++){ // jth sample
        //if(i == 0)
        //  printf("sample_X_%d = ", j);
        // on each covariates
        float min_sim = 99999.0f;
        for(unsigned int k = 0; k < ncols_X; k++){ // kth variable
            float tmp_sim = 0.0f;
            int msrlevel = MSR_LEVELS[k];
            float X_ik = X[i * ncols_X + k];
            float sample_X_jk = sample_X[j * ncols_X + k];

            //if(i == 0)
            //  printf("%f ", sample_X_jk);

            if(msrlevel == 0 || msrlevel == 1){
                if(X_ik == sample_X_jk)
                    tmp_sim = 1.0f;
                else
                    tmp_sim = 0.0f;
            }
            else{
                float SD_k = SD_evs[k];
                float sum_sqr = 0.0f;
                for(unsigned int m = 0; m < nrows_evs; m++){
                    float evs_mk = evs[m * ncols_X + k];
                    float diff = evs_mk - sample_X_jk;
                    sum_sqr += diff * diff;
                }
                float SD_kj = sqrt(sum_sqr/nrows_evs);
                //if(i == 0)
                //  printf("SD_evj = %f ", SD_kj);
                float diff = (X_ik - sample_X_jk);
                float denom = SD_k * SD_k / SD_kj;
                tmp_sim = exp(-0.5f *  (diff * diff) / (denom * denom));
            }
            //if(i == 0)
            //  printf("%f ", tmp_sim);
            if(tmp_sim < min_sim){
                min_sim = tmp_sim;
            }
        }
        //if(i == 0)
        //  printf("\n");
        similarities[j] = min_sim;
    }

    // now do prediction
    if(mode == 1){ // predict class
        int max_sim_idx = 0;
        float max_sim = -9999.0f;
        float max_w = -9999.0f;
        for(unsigned int j = 0; j < nrows_samples; j++){
            if(sample_weights[j] > max_w){
                max_w = sample_weights[j];
            }
            float sim = similarities[j] * sample_weights[j];
            if(sim > max_sim){
                max_sim = sim;
                max_sim_idx = j;
            }
        }
        X_predictions[i] = sample_attributes[max_sim_idx];
        X_uncertainties[i] = 1.0f - max_sim / max_w;
    }
    else{ // predict property
        float sum_sim = 0.0f;
        float max_sim = -9999.0f;
        float sum_weighted_property = 0.0f;
        float max_w = -9999.0f;
        for(unsigned int j = 0; j < nrows_samples; j++){
            if(sample_weights[j] > max_w){
                max_w = sample_weights[j];
            }
            float sim = similarities[j] * sample_weights[j];
            if(sim > max_sim){
                max_sim = sim;
            }
            if(sim >= 1.0f - threshold){
                sum_sim += sim;
                sum_weighted_property += sim * sample_attributes[j];
            }
        }
        //if(sum_sim == 0.0) xxx; //possibly occur if there are categorical variables
        X_predictions[i] = sum_weighted_property / sum_sim;
        X_uncertainties[i] = 1.0f - max_sim / max_w;
    }

}

__kernel void iPSM_Predict_Sequential_naive(const unsigned int nrows_evs, const unsigned int nrows_X, const unsigned int ncols_X, const unsigned int nrows_samples,
                           const unsigned int mode, const float threshold, __global int* MSR_LEVELS, __global float* evs, __global float* SD_evs,
                           __global float* X, __global float* sample_X, __global float* sample_weights, __global float* sample_attributes,
                           __global float* X_predictions, __global float* X_uncertainties)
{
    for(unsigned int i = 0; i < nrows_X; i++){
      // compute the similarity btw ith location to each of the samples
      float similarities[N_SAMPLES];
      for(unsigned int j = 0; j < nrows_samples; j++){ // jth sample
          //if(i == 0)
          //  printf("sample_X_%d = ", j);
          // on each covariates
          float min_sim = 99999.0f;
          for(unsigned int k = 0; k < ncols_X; k++){ // kth variable
              float tmp_sim = 0.0f;
              int msrlevel = MSR_LEVELS[k];
              float X_ik = X[i * ncols_X + k];
              float sample_X_jk = sample_X[j * ncols_X + k];

              //if(i == 0)
              //  printf("%f ", sample_X_jk);

              if(msrlevel == 0 || msrlevel == 1){
                  if(X_ik == sample_X_jk)
                      tmp_sim = 1.0f;
                  else
                      tmp_sim = 0.0f;
              }
              else{
                  float SD_k = SD_evs[k];
                  float sum_sqr = 0.0f;
                  for(unsigned int m = 0; m < nrows_evs; m++){
                      float evs_mk = evs[m * ncols_X + k];
                      float diff = evs_mk - sample_X_jk;
                      sum_sqr += diff * diff;
                  }
                  float SD_kj = sqrt(sum_sqr/nrows_evs);
                  //if(i == 0)
                  //  printf("SD_evj = %f ", SD_kj);
                  float diff = (X_ik - sample_X_jk);
                  float denom = SD_k * SD_k / SD_kj;
                  tmp_sim = exp(-0.5f *  (diff * diff) / (denom * denom));
              }
              //if(i == 0)
              //  printf("%f ", tmp_sim);
              if(tmp_sim < min_sim){
                  min_sim = tmp_sim;
              }
          }
          //if(i == 0)
          //  printf("\n");
          similarities[j] = min_sim;
      }

      // now do prediction
      if(mode == 1){ // predict class
          int max_sim_idx = 0;
          float max_sim = -9999.0f;
          float max_w = -9999.0f;
          for(unsigned int j = 0; j < nrows_samples; j++){
              if(sample_weights[j] > max_w){
                  max_w = sample_weights[j];
              }
              float sim = similarities[j] * sample_weights[j];
              if(sim > max_sim){
                  max_sim = sim;
                  max_sim_idx = j;
              }
          }
          X_predictions[i] = sample_attributes[max_sim_idx];
          X_uncertainties[i] = 1.0f - max_sim / max_w;
      }
      else{ // predict property
          float sum_sim = 0.0f;
          float max_sim = -9999.0f;
          float sum_weighted_property = 0.0f;
          float max_w = -9999.0f;
          for(unsigned int j = 0; j < nrows_samples; j++){
              if(sample_weights[j] > max_w){
                  max_w = sample_weights[j];
              }
              float sim = similarities[j] * sample_weights[j];
              if(sim > max_sim){
                  max_sim = sim;
              }
              if(sim >= 1.0f - threshold){
                  sum_sim += sim;
                  sum_weighted_property += sim * sample_attributes[j];
              }
          }

          X_predictions[i] = sum_weighted_property / sum_sim;
          X_uncertainties[i] = 1.0f - max_sim / max_w;
      }
  }

}
