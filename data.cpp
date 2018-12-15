#include <fstream>  //for mnist load
class DatasetMnist
{
    public:
        static const int width = 28;
        static const int height = 28;
        static const int classes  = 10;
        int sample[width][height];
    private:
        ifstream image;
        ifstream label;
    public:
        DatasetMnist()
        {
            //MnistÊı¾İÏÂÔØ: http://yann.lecun.com/exdb/mnist/
            const string training_image_fn = "mnist/train-images.idx3-ubyte";
            const string training_label_fn = "mnist/train-labels.idx1-ubyte";
            image.open(training_image_fn.c_str(), ios::in | ios::binary);
            label.open(training_label_fn.c_str(), ios::in | ios::binary );
            char number;
            for (int i = 0; i < 16; i++)
            {
                image.read(&number, sizeof(char));
            }
            for (int i = 0; i < 8; i++)
            {
                label.read(&number, sizeof(char));
            }
        };
        ~DatasetMnist()
        {
            image.close();
            label.close();
        };
        int load(double* input, double* target)  //TODO: mini-batch
        {
            char number;
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++)
                {
                    image.read(&number, sizeof(char));
                    if (number == 0)
                    {
                        sample[i][j] = 0; 
                    } 
                    else
                    {
                        sample[i][j] = 1;
                    }
                }
            }
            for (int j = 0; j < height; j++)   //matrix to vector
            {
                for (int i = 0; i < width; i++)
                {
                    int pos = i + (j - 1) * width;
                    input[pos] = sample[i][j];
                }
            }
            if(target != NULL)  //for inference mode, no target
            {
                label.read(&number, sizeof(char));
                for (int i = 0; i < classes; ++i)
                {
                    target[i] = 0.0;
                }
                target[number] = 1.0;
            }
            if(0)   //show mnist image
            {
                cout << "Image:" << (int)(number) << endl;
                for (int j = 0; j < height; j++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        cout << sample[i][j];
                    }
                    cout << endl;
                }
            }
            return (int)(number);
        };
};
class DatasetXOR
{
    private:
        int samples[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
        int labels[4][1] = {{0},   {1},   {1},   {0}};
    public:
        static const int width = 2;
        static const int height = 1;
        static const int classes  = 1;
    public:
        DatasetXOR()
        {
        };
        ~DatasetXOR()
        {
        };
        int load(double* input, double* target)
        {
            int index = rand() % 4;
            int* sample = samples[index];
            input[0] = sample[0];
            input[1] = sample[1];
            int* label = labels[index];
            target[0] = label[0];
            return target[0];
        };
};
