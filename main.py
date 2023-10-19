from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, AUC, RootMeanSquaredError
from sklearn.metrics import f1_score

from tensorflow.python.keras.callbacks import EarlyStopping

import os

import utilities
# Generate data paths with labels
class Application:
    def __init__(self) -> None:
        self.data_handler = utilities.HandleData()
        self.visualizer = utilities.Visualization()
        self.model_handler = utilities.HandleModel()
        self.DATA_DIR = 'data'
        self.SAMPLE_SIZE=5000
        self.TEST_BATCH_SIZE=50  
        self.TEST_STEPS=50
        
    def main(self):
        df = self.data_handler.load_data(self.DATA_DIR)
        df = self.data_handler.balance_dataset(df, self.SAMPLE_SIZE)

        train_df, test_df, valid_df = self.data_handler.split_data(df)
        train_generator, test_generator, valid_generator = self.data_handler.create_data_generators(train_df, test_df, valid_df)
        classes=list(train_generator.class_indices.keys())
        
        # self.visualizer.show_image_samples(train_generator)
        num_classes=len(classes)
        model = self.model_handler.cerate_model(num_classes)
        model.compile(Adam(learning_rate=.001), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC(), 'mae', 'mse', RootMeanSquaredError()])
        model.summary()

        model_name = 'Xception+MobileNet'
        epochs = 1
        history = self.model_handler.train_model(train_generator, valid_generator, model, model_name, epochs)

        self.visualizer.plot_training(history)

        save_dir=r'./'
        subject='best'
        acc=model.evaluate( test_generator, batch_size=self.TEST_BATCH_SIZE, verbose=1, steps=self.TEST_STEPS, return_dict=False)[1]*100
        msg=f'accuracy on the test set is {acc:5.2f} %'
        self.visualizer.print_in_color(msg, (0,255,0),(55,65,80))
        save_id=str (model_name +  '-' + subject +'-'+ str(acc)[:str(acc).rfind('.')+3] + '.h5')
        save_loc=os.path.join(save_dir, save_id)
        model.save(save_loc)

        print_code=0
        predictions=model.predict(test_generator) 
        self.visualizer.print_info( test_generator, predictions, print_code, save_dir, subject ) 

if __name__ == "__main__":
    app = Application()
    app.main()