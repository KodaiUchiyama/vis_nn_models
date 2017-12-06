# coding:utf-8
'''
Lamtharn (Hanoi) Hantrakul
29/4/2017

=====================
HYPERPARAMETER SEARCH
=====================
This is a controller for doing hyperparameter search over models used in the project.

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from exp_models.CNN_LSTM_models import *  # Contains the models we are testing
from model_utils import *  # Contains some useful functions for loading datasets
from keras.models import load_model

#########################
#Set data_name
train_data_name = 'vis_train_dataX_dataY.h5'  # Set to name of h5py dataset
test_data_name = 'vis_test_dataX_dataY.h5'
"""
Main function call for doing random hyperparameter search
ハイパパラメータ生成、種はランダム
"""
def main():
    # 繰り返し回数
    num_trials = 1


    final_accuracies = []
    learning_rates = []
    weight_scales = []

    for i in range(num_trials):

        # random search over logarithmic space of hyperparameters
        #lr = 10**np.random.uniform(-3.0,-6.0)
        lr = 0.1
        #ws = 10**np.random.uniform(-2.0,-4.0)
        ws = 0.01
        #learning_rates.append(lr)
        #weight_scales.append(ws)

        """Read a sample from h5py dataset and return key dimensions
model_utils内の関数
            Example returns:
                frame_h = 90
                frame_w = 160
                channels = 15
                audio_vector_dim = 42 = dataY_sample.shape[0]
        """
        frame_h, frame_w, channels, audio_vector_dim = returnH5PYDatasetDims(data_name='vis_train_dataX_dataY.h5')

        """Load full dataset as an HDF5 matrix object for use in Keras model

                Example returns and corresponding matrix shapes:
                    dataX_train.shape = (26000,90,160,15)
                    dataY_train.shape = (26000,42)
                    dataX_test.shape = (4000,90,160,15)
                    dataY_test.shape = (4000,42)
        """
        dataX_train, dataY_train = load5hpyTrainData(data_name = 'vis_train_dataX_dataY.h5')
        dataX_test, dataY_test = load5hpyTestData(data_name = 'vis_test_dataX_dataY.h5')

        # create defined model with given hyper parameters
        model = CNN_LSTM_model(image_dim=(frame_h,frame_w,channels),
                               audio_vector_dim=audio_vector_dim,
                               learning_rate=lr,
                               weight_init=ws)

        # load custom callbacksカスタムコールバック!?あとでサーベイ
        #loss_history = LossHistory()
        #acc_history = AccuracyHistory()
        #callbacks_list = [loss_history, acc_history]

        # train the model
        fit=model.fit(dataX_train,
                  dataY_train,
                  shuffle='batch',
                  epochs=5,
                  batch_size=1000,  # 10000 is the maximum number of samples that fits on TITANX 12GB Memory
                  validation_data=(dataX_test, dataY_test),
                  verbose=1)
                  #callbacks = callbacks_list)
        model.save('my_model.h5')
        #model = load_model('my_model.h5')
        #可視化
        # フォルダの作成
        # make output directory
        folder = 'results/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        def plot_history_loss(fit):
            plt.plot(fit.history['loss'],label="loss for training")
            plt.plot(fit.history['val_loss'],label="loss for validation")
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='upper right')

        plot_history_loss(fit)
        plt.savefig(folder + '/vis_loss.pdf')
        '''
        # Graph training history
        plotAndSaveData(loss_history, "Loss", learning_rate_hp=lr, weight_hp=ws, title="Loss History")
        plotAndSaveData(acc_history, "Accuracy", learning_rate_hp=lr, weight_hp=ws, title="Acc History")

        # Save final accuracy 最後のplotAndSaveSessionで使用
        final_accuracies.append(acc_history.test_acc[-1])
        '''
        '''これは1動画が入っているテストサンプルを用意してから、loadmatとかでいいでしょh5py経由しなくても
        # Make a directory to store the predicted spectrums
        folder_name = '{lr:%s}-{ws:%s}' % (str(lr), str(ws))
        dir_path = makeDir(folder_name)  # dir_path = "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
        # Run the model on some unseen data and save the predicted spectrums in the directory defined previously

        genAndSavePredSpectrum(model,
                               dir_path,
                               window_length = 300,
                               data_name=data_name)
        '''
    #Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
    #ランダムに重みと学習率を生成して、三次元のグラフを作って、どのパラメータの値が効果的かを目視できる
    #plotAndSaveSession(learning_rates,weight_scales,final_accuracies)
    print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")

if __name__ == '__main__':
    main()
