/*
  train: mini-batch, adam*-optimizer,
  loss: cross_entropy
  initialize: orthogonal, identity, glorot_*, lecun_*, he_*, yue_*
  non-linear: activation: ReLU, softmax, 研究点: 可训练分段线性激活函数
  miscs: 研究点: loss_landspace_visualization(Python实现了:loss_surface.py
  ---
  structure: deep, cnn, rnn, ae, gan
  algorithm: unsupervised, reinforcement
*/
#include <iostream>

using namespace std;

class DataVector
{
    public:
        double* data;
        DataVector(int size, bool initialize_with_minimal)
        {
          data = new double[size];
          for(int i=0;i<size;i++)
          {
            if(initialize_with_minimal)  //TODO  初始化应该单独做成一个类来实现不同初始化策略
            {
              data[i]=(double)(rand()%100)/1000.0-0.05;  //坑: .0
            }
            else
            {
              data[i]=0.0;
            }
          }
        }
        ~DataVector()
        {
          delete[] data;
        }
};
class DataMatrix
{
    private:
        int size1;
    public:
        double** data;
        DataMatrix(int size1, int size2, bool initialize_with_minimal)
        {
          this->size1 = size1;
          data = new double*[size1];
          for(int i=0;i<size1;i++)
          {
            data[i] = new double[size2];
            for(int j=0;j<size2;j++)
            {
              if(initialize_with_minimal)  //TODO  初始化应该单独做成一个类来实现不同初始化策略
              {
                data[i][j]=(double)(rand()%100)/200.0-0.25;  //坑: .0
              }
              else
              {
                data[i][j]=0.0;
              }
            }
          }
        }
        ~DataMatrix()
        {
          for(int i=0;i<size1;i++)
          {
            delete[] data[i];
          }
          delete[] data;
        }
};
class Blas
{
    public:
        static void vector_cmul_matrix(int vector_size, int matrix_size, double* vector, double** matrix, double* output)  //儿子,static,学习定义和使用
        {
          for (int i = 0;i < vector_size;i++)
          {
            for (int j = 0;j < matrix_size;j++)
            {
              output[j] += vector[i] * matrix[i][j];
            }
          }
        }
        static void vector_add_vector(int vector_size, double* vector_a, double* vector_b, double* output)
        {
          for(int i=0;i<vector_size;i++)
          {
            output[i]=vector_a[i] + vector_b[i];
          }
        }
        static double vector_mulsum_vector(int vector_size, double* vector_a, double* vector_b)
        {
          double sum = 0.0;
          for (int i = 0; i < vector_size; i++)
          {
            sum += vector_a[i] * vector_b[i];
          }
          return sum;
        }
        static void vector_mul_scalar(int vector_size, double* vector, double scalar, double* output)
        {
          for(int i=0;i<vector_size;i++)
          {
            output[i]=vector[i] * scalar;
          }
        }
        static void vector_zero(int vector_size, double* vector)
        {
          for(int i=0;i<vector_size;i++)
          {
            vector[i] = 0.0;
          }
        }
};
class Activation
{
    public:
        static double activation_tanh_other(double x)
        { 
          double t = exp(-2.0 * x);
          double y = (1.0 - t) / (1.0 + t);
          return y;
        }
        static double activation_tanh(double x)
        { 
          double a = exp(x);
          double b = exp(-x);
          double y = (a - b) / (a + b);
          return y;
        }
        static double activation_derivate_tanh(double x)
        {
          double y = 1.0 - (x * x);
          return y;
        }
        static double activation_sigmoid(double x)
        {
          double y = 1.0 / (1.0 + exp(-x));
          return y;
        }
        static double activation_derivate_sigmoid(double x)
        {
          double y = x * (1.0 - x);
          return y;
        }
};
class Loss
{
    public:
        static double square_error(int size, double* output, double* target)   //损失函数loss的一种,square_error
        {
          double loss = 0.0;
          for (int i = 0; i < size; i++)
          {
            loss += (output[i] - target[i]) * (output[i] - target[i]);             //相减计算差距,平方去掉符号
          }
          loss *= 0.5;                                                             //*1/2在计算导数的时候和^2过来的抵消                                             
          return loss;
        }    
};
class Network
{
  private:
    int input_size;
    int hidden_size;
    int output_size;
  private:  //just for train
    DataVector* theta_oh;    //diff(output和target的差值),乘以导数 (用y=x^2做例子来理解)
    DataMatrix* delta_oh;    //delta: 是theta每个x相乘算得的,每个weight要调整的值
    DataVector* theta_hi;    //sum(权重*theta_oh的代数和,链式法则),乘以导数 (用y=x^2做例子来理解)
    DataMatrix* delta_hi;    //delta: 是theta每个x相乘算得的,每个weight要调整的值
    DataMatrix* weight_ih;
    DataVector* bias_ih;
    DataVector* hidden;    //可以局部变量         
    DataMatrix* weight_ho;
    DataVector* bias_ho;
    DataVector* output;
  public:
    DataVector* input;
    DataVector* target;
  private:
    double activation_nolinear(double x)
    { 
      return Activation::activation_tanh(x);
    }
    double activation_derivate_nolinear(double x)
    { 
      return Activation::activation_derivate_tanh(x);
    }
  public:
    Network(int input_size, int hidden_size, int output_size)
    {
      this->input_size = input_size;
      this->hidden_size = hidden_size;
      this->output_size = output_size;
      input = new DataVector(input_size, true);
      weight_ih = new DataMatrix(input_size, hidden_size, true);
      bias_ih = new DataVector(hidden_size, true);
      hidden = new DataVector(hidden_size,false); 
      weight_ho = new DataMatrix(hidden_size, output_size, true);
      bias_ho = new DataVector(output_size, true);
      output = new DataVector(output_size, true);
      target = new DataVector(output_size, true);
    }
    ~Network()
    {
      delete input;
      delete weight_ih;
      delete bias_ih;
      delete hidden;
      delete weight_ho;
      delete bias_ho;
      delete output;
      delete target;      
      // just for train
      if(delta_oh != NULL) { delete delta_oh; }
      if(theta_oh != NULL) { delete theta_oh; }
      if(delta_hi != NULL) { delete delta_hi; }
      if(theta_hi != NULL) { delete theta_hi; }
    }
    void train_init()
    {
      delta_oh = new DataMatrix(hidden_size, output_size, false);
      theta_oh = new DataVector(output_size, false);
      delta_hi = new DataMatrix(input_size, hidden_size, false);
      theta_hi = new DataVector(hidden_size, false);           
    }
    void forward_propagation()
    {
      Blas::vector_zero(hidden_size, hidden->data);
      Blas::vector_zero(output_size, output->data);
      Blas::vector_cmul_matrix(input_size, hidden_size, input->data, weight_ih->data, hidden->data);  //儿子,static
      Blas::vector_add_vector(hidden_size, hidden->data, bias_ih->data, hidden->data);
      for(int j=0;j<hidden_size;j++)
      {
        hidden->data[j]=activation_nolinear(hidden->data[j]);
      }
      Blas::vector_cmul_matrix(hidden_size,output_size,hidden->data,weight_ho->data,output->data);
      Blas::vector_add_vector(output_size, output->data, bias_ho->data, output->data);      
      for(int j=0;j<output_size;j++)
      {
        output->data[j]=activation_nolinear(output->data[j]);
      }
    }
    void backward_propagation(double learning_rate, double momentum)
    {
      for(int j=0;j<output_size;j++)
      {
        double distance = target->data[j] - output->data[j];     //To Ting: 注意顺序, 影响到符号     输出和目标的差值   0.38 - 0.2 = 0.18   (hidden = 0.8    w = 0.5    tanh = O = 0.38)
        double derivate = activation_derivate_nolinear(output->data[j]);   //对output算tanh的导数   deri_tanh(0.38) = 1 - 0.1444 = 0.8556
        theta_oh->data[j] = distance * derivate;   //直接按照距离和导数计算出来欲调节的值 (可以实现为局部变量)     //0.18 * 0.8556 ~= 0.154 
      }
      for (int i = 0; i < hidden_size; i++)
      {
        for (int j = 0; j < output_size; j++)
        {
          // = (theta_oh[j] * hidden[i])                                                 //优化:无;               问题:步子太大       w += (0.154 * 0.8 ~= 0.123)
          // = (learning_rate * theta_oh[j] * hidden[i])                                 //优化:小步小步的探;     问题:碰到局部极小值小坑就翻不过去
          delta_oh->data[i][j] = (learning_rate * theta_oh->data[j] * hidden->data[i]) + (momentum * delta_oh->data[i][j]);    //优化:历史下降惯性;     问题:好多了 (更牛的优化器,继续研究)
          weight_ho->data[i][j] += delta_oh->data[i][j];                                                                           
        }
      }
      Blas::vector_mul_scalar(output_size, theta_oh->data, learning_rate, bias_ho->data);  //问题:bias需要使用momentum吗?   +=还是等于呢?
      for (int i = 0; i < hidden_size; i++)    //隐藏层的梯度
      {
        double sum = Blas::vector_mulsum_vector(output_size,weight_ho->data[i],theta_oh->data);  //TODO：sum需在 weight_ho更新前吗？好像不是。
        double derivate = activation_derivate_nolinear(hidden->data[i]);
        theta_hi->data[i] = sum * derivate;  //sigmoid derive: n * (1-n)代表最小化下山的方向(中间sigmoid);    
      }
      for (int i = 0; i < input_size; i++)
      {
        for (int j = 0 ; j < hidden_size ; j++ )
        {
          delta_hi->data[i][j] = (learning_rate * theta_hi->data[j] * input->data[i]) + (momentum * delta_hi->data[i][j]);   //要调节的值
          weight_ih->data[i][j] += delta_hi->data[i][j];                                                            //实际调节了                                                            
        }
      }
      Blas::vector_mul_scalar(hidden_size, theta_hi->data, learning_rate, bias_ih->data);  //问题:bias需要使用momentum吗?   +=还是等于呢?
    } 
    double loss()
    {
      return Loss::square_error(output_size, output->data, target->data);
    }
};

#include "data.cpp"
void main(int argc, char** argv)
{
  srand(123);    //#include <time.h>    time_t t; srand((unsigned)time(&t));
  DatasetMnist dataset;  //DatasetXOR dataset;
  int input_size = dataset.width*dataset.height;  //from Dataset
  int hidden_size = 128;   //by design
  int output_size = dataset.classes;   //from Dataset
  Network network(input_size, hidden_size, output_size);
  network.train_init();
  for(int epoch=0;epoch<60000;epoch++)
  {
    dataset.load(network.input->data, network.target->data);  //for(int i=0;i<input_size;i++) { network.input[i] = i % 3 - 1.0; }   //牛: % -
    for(int repeat=0; repeat<4; repeat++)
    {
      network.forward_propagation();
      network.backward_propagation(0.0001, 0.9);
    }
    if(epoch % 100 == 0)
    {
      cout<<"loss="<<network.loss()<<endl;
    }
  }   
}
