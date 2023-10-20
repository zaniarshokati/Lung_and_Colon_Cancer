import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, MobileNet
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Dropout,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Add,
    AveragePooling2D,
    Input,
    ZeroPadding2D,
    Concatenate,
)
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.python.keras.callbacks import EarlyStopping


class HandleData:
    def __init__(self) -> None:
        self.train_size = 0.8
        self.test_size = 0.1
        self.data = []

    def load_data(self, DATA_DIR):
        for root, dirs, files in os.walk(DATA_DIR):
            for cls in dirs:
                if cls not in ["lung_image_sets", "colon_image_sets"]:
                    class_path = os.path.join(root, cls)
                    for f in os.listdir(class_path):
                        self.data.append(
                            {"filepaths": os.path.join(class_path, f), "labels": cls}
                        )

        df = pd.DataFrame(self.data)
        # print(df['labels'].value_counts())
        return df

    def balance_dataset(self, df, sample_size):
        sample_list = []
        group = df.groupby("labels")
        sample_list = [
            group.get_group(label).sample(
                sample_size, replace=False, random_state=123, axis=0
            )
            for label in df["labels"].unique()
        ]
        df = pd.concat(sample_list, axis=0).reset_index(drop=True)
        # print(len(df))
        return df

    def split_data(self, df):
        train_df, test_valid_df = train_test_split(
            df, train_size=self.train_size, shuffle=True, random_state=123
        )
        test_df, valid_df = train_test_split(
            test_valid_df,
            train_size=self.test_size / (1 - self.train_size),
            shuffle=True,
            random_state=123,
        )
        # print('train_df length:', len(train_df), 'test_df length:', len(test_df), 'valid_df length:', len(valid_df))
        return train_df, test_df, valid_df

    def create_data_generators(
        self,
        train_df,
        test_df,
        valid_df,
        img_size=(224, 224),
        batch_size=32,
        test_batch_size=50,
    ):
        # Custom preprocessing function to scale pixel values between -1 and +1
        def scale_image(img):
            return img / 127.5 - 1

        # Data augmentation and preprocessing
        datagen = ImageDataGenerator(
            preprocessing_function=scale_image,
        )

        # Create data generators
        train_generator = datagen.flow_from_dataframe(
            train_df,
            x_col="filepaths",
            y_col="labels",
            target_size=img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size,
        )

        test_generator = datagen.flow_from_dataframe(
            test_df,
            x_col="filepaths",
            y_col="labels",
            target_size=img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=False,  # No shuffling for testing data
            batch_size=test_batch_size,  # Test the entire dataset in one batch
        )

        valid_generator = datagen.flow_from_dataframe(
            valid_df,
            x_col="filepaths",
            y_col="labels",
            target_size=img_size,
            class_mode="categorical",
            color_mode="rgb",
            shuffle=True,
            batch_size=batch_size,
        )

        return train_generator, test_generator, valid_generator


class Visualization:
    def __init__(self) -> None:
        pass

    def show_image_samples(self, train_generator, max_images=25):
        test_dict = train_generator.class_indices
        classes = list(test_dict.keys())

        # Get a sample batch from the generator
        images, labels = next(train_generator)

        num_samples = len(labels)
        num_rows = min(int(np.ceil(num_samples / 5)), 5)  # Maximum 25 images (5x5 grid)

        plt.figure(figsize=(15, 15))

        for i in range(min(max_images, num_samples)):
            plt.subplot(num_rows, 5, i + 1)
            image = (images[i] + 1) / 2  # Scale images between 0 and 1
            plt.imshow(image)

            index = np.argmax(labels[i])
            class_name = classes[index]
            plt.title(class_name, color="blue", fontsize=12)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_training(self, tr_data):
        # Extract training and validation data
        tacc = tr_data.history["accuracy"]
        tloss = tr_data.history["loss"]
        vacc = tr_data.history["val_accuracy"]
        vloss = tr_data.history["val_loss"]

        # Get the number of epochs directly
        Epochs = range(1, len(tacc) + 1)

        # Find the epoch with the lowest validation loss and the highest validation accuracy
        index_loss = vloss.index(min(vloss)) + 1
        val_lowest = min(vloss)
        index_acc = vacc.index(max(vacc)) + 1
        acc_highest = max(vacc)

        # Plot the data
        plt.style.use("fivethirtyeight")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        # Plot training and validation loss
        sc_label = f"best epoch = {index_loss}"
        axes[0].plot(Epochs, tloss, "r", label="Training loss")
        axes[0].plot(Epochs, vloss, "g", label="Validation loss")
        axes[0].scatter(index_loss, val_lowest, s=150, c="blue", label=sc_label)
        axes[0].set_title("Training and Validation Loss")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # Plot training and validation accuracy
        vc_label = f"best epoch = {index_acc}"
        axes[1].plot(Epochs, tacc, "r", label="Training Accuracy")
        axes[1].plot(Epochs, vacc, "g", label="Validation Accuracy")
        axes[1].scatter(index_acc, acc_highest, s=150, c="blue", label=vc_label)
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()  # Properly adjust subplot layout
        plt.show()

    def print_info(self, test_gen, preds, print_code, save_dir, subject):
        class_dict = test_gen.class_indices
        labels = test_gen.labels
        file_names = test_gen.filenames
        new_dict = {value: key for key, value in class_dict.items()}
        classes = list(new_dict.values())
        errors = 0
        error_list = []
        true_class = []
        pred_class = []
        prob_list = []
        error_indices = []
        y_pred = []

        for i, p in enumerate(preds):
            pred_index = np.argmax(p)
            true_index = labels[i]

            if pred_index != true_index:
                errors += 1
                error_list.append(file_names[i])
                true_class.append(new_dict[true_index])
                pred_class.append(new_dict[pred_index])
                prob_list.append(p[pred_index])
                error_indices.append(true_index)
            y_pred.append(pred_index)

        if print_code != 0:
            if errors > 0:
                r = min(print_code, errors)
                header = "{0:^28s}{1:^28s}{2:^28s}{3:^16s}".format(
                    "Filename", "Predicted Class", "True Class", "Probability"
                )
                self.print_in_color(header, (0, 255, 0), (55, 65, 80))

                for i in range(r):
                    split1, fname = os.path.split(error_list[i])
                    split2, split1 = os.path.split(split1)
                    filename = f"{split1}/{fname}"
                    msg = f"{filename:^28s}{pred_class[i]:^28s}{true_class[i]:^28s}    {prob_list[i]:.4f}"
                    self.print_in_color(msg, (255, 255, 255), (55, 65, 60))
            else:
                msg = "With accuracy of 100%, there are no errors to print"
                self.print_in_color(msg, (0, 255, 0), (55, 65, 80))

        if errors > 0:
            plot_bar = []
            plot_class = []

            for key, value in new_dict.items():
                count = error_indices.count(key)
                if count != 0:
                    plot_bar.append(count)
                    plot_class.append(value)

            fig = plt.figure(figsize=(10, len(plot_class) / 3))
            plt.style.use("fivethirtyeight")

            for i in range(len(plot_class)):
                c, x = plot_class[i], plot_bar[i]
                plt.barh(c, x)

            plt.title("Errors by Class on Test Set")

        y_true = np.array(labels)
        y_pred = np.array(y_pred)

        if len(classes) <= 30:
            cm = confusion_matrix(y_true, y_pred)
            length = len(classes)

            if length < 8:
                fig_width, fig_height = 8, 8
            else:
                fig_width, fig_height = int(length * 0.5), int(length * 0.5)

            plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(cm, annot=True, vmin=0, fmt="g", cmap="Blues", cbar=False)
            plt.xticks(np.arange(length) + 0.5, classes, rotation=90)
            plt.yticks(np.arange(length) + 0.5, classes, rotation=0)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.show()

        clr = classification_report(y_true, y_pred, target_names=classes)
        print("Classification Report:\n----------------------\n", clr)

    def print_in_color(self, text, foreground_color, background_color):
        # Extract RGB components for the foreground and background colors
        r_fg, g_fg, b_fg = foreground_color
        r_bg, g_bg, b_bg = background_color

        # Define the color format using ANSI escape codes
        color_format = f"\33[38;2;{r_fg};{g_fg};{b_fg};48;2;{r_bg};{g_bg};{b_bg}m"

        # Print the text with the specified color format
        print(color_format + text)

        # Reset the color to default (black text on a white background)
        print("\33[0m")
        return


class HandleModel:
    def cerate_model(self, num_classes):
        mobilenet = MobileNet(weights="imagenet", include_top=False)
        xception = Xception(weights="imagenet", include_top=False)

        # freeze the layers
        for layer in mobilenet.layers:
            layer.trainable = False
        for layer in xception.layers:
            layer.trainable = False

        # create a single input layer
        input_layer = Input(shape=(224, 224, 3))

        # connect the input layer to both models
        x1 = mobilenet(input_layer)
        x2 = xception(input_layer)

        # concatenate the outputs from both models
        concatenated = Concatenate()([x1, x2])

        # add a few dense layers for the final prediction
        flat = Flatten()(concatenated)
        dense = Dense(units=1024, activation="relu")(flat)
        dense = Dense(512, activation="relu")(dense)
        dense = Dense(units=256, activation="relu")(dense)
        predictions = Dense(num_classes, activation="softmax")(dense)
        # compile the model
        # this is the model we will train
        model = Model(inputs=input_layer, outputs=predictions)
        return model

    def train_model(self, train_generator, valid_generator, model, model_name, epochs):
        early_stop = EarlyStopping(
            monitor="val_loss", patience=15, mode="min", verbose=1
        )
        # Create a list of callbacks for training
        callbacks = [
            LearningRateAdjustment(
                model=model,
                patience=9,
                stop_patience=3,
                threshold=0.9,
                factor=0.5,
                dwell=True,
                model_name=model_name,
                freeze=False,
                initial_epoch=0,
            ),
            early_stop,
        ]

        # Set the total number of epochs for printing in the LearningRateAdjustment callback
        LearningRateAdjustment.tepochs = epochs

        # Train the model
        history = model.fit(
            x=train_generator,  # Training data
            epochs=epochs,  # Number of training epochs
            callbacks=callbacks,  # Callbacks for dynamic learning rate adjustment
            verbose=1,  # Verbosity level
            validation_data=valid_generator,  # Validation data
            validation_steps=None,  # Number of validation steps
            shuffle=False,  # Disable shuffling of training data
            initial_epoch=0,  # Initial epoch
        )

        return history


class LearningRateAdjustment(keras.callbacks.Callback):
    def __init__(
        self,
        model,
        patience,
        stop_patience,
        threshold,
        factor,
        dwell,
        model_name,
        freeze,
        initial_epoch,
    ):
        super(LearningRateAdjustment, self).__init__()

        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.lr = float(tf.keras.backend.get_value(model.optimizer.lr))
        self.highest_tracc = 0.0
        self.lowest_vloss = np.inf
        self.initial_epoch = initial_epoch
        self.best_weights = self.model.get_weights()

        # Print a message indicating the start of training
        msg = f"Starting training using base model {model_name} with {'frozen' if freeze else 'trainable'} layers.\n"
        self.print_in_color(msg, (244, 252, 3), (55, 65, 80))

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the duration of the current epoch
        later = time.time()
        duration = later - self.now

        # Print a header if it's the initial epoch or there's a reset
        if epoch == self.initial_epoch or LearningRateAdjustment.reset:
            LearningRateAdjustment.reset = False
            msg = "{0:^8s}{1:^10s}{2:^9s}{3:^9s}{4:^9s}{5:^9s}{6:^9s}{7:^11s}{8:^8s}".format(
                "Epoch",
                "Loss",
                "Accuracy",
                "V_loss",
                "V_acc",
                "LR",
                "Next LR",
                "Monitor",
                "Duration",
            )
            self.print_in_color(msg, (244, 252, 3), (55, 65, 80))

        # Get current learning rate, validation loss, and accuracy
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        current_lr = lr
        v_loss = logs.get("val_loss")
        acc = logs.get("accuracy")
        v_acc = logs.get("val_accuracy")
        loss = logs.get("loss")

        monitor = "accuracy" if acc < self.threshold else "val_loss"

        if acc < self.threshold:
            if acc > self.highest_tracc:
                self.highest_tracc = acc
                LearningRateAdjustment.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                if v_loss < self.lowest_vloss:
                    self.lowest_vloss = v_loss
                color = (0, 255, 0)
                self.lr = lr
            else:
                if self.count >= self.patience - 1:
                    color = (245, 170, 66)
                    self.lr = lr * self.factor
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
                    self.count = 0
                    self.stop_count += 1
                    if self.dwell:
                        self.model.set_weights(LearningRateAdjustment.best_weights)
                    elif v_loss < self.lowest_vloss:
                        self.lowest_vloss = v_loss
                else:
                    self.count += 1
        else:
            if v_loss < self.lowest_vloss:
                self.lowest_vloss = v_loss
                LearningRateAdjustment.best_weights = self.model.get_weights()
                self.count = 0
                self.stop_count = 0
                color = (0, 255, 0)
                self.lr = lr
            else:
                if self.count >= self.patience - 1:
                    color = (245, 170, 66)
                    self.lr = self.lr * self.factor
                    self.stop_count += 1
                    self.count = 0
                    tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)
                    if self.dwell:
                        self.model.set_weights(LearningRateAdjustment.best_weights)
                else:
                    self.count += 1

            # if acc > self.highest_tracc:
            #     self.highest_tracc = acc
            msg = f"{str(epoch + 1):^3s}/{str(LearningRateAdjustment.tepochs):4s} {loss:^9.3f}{acc*100:^9.3f}{v_loss:^9.5f}{v_acc*100:^9.3f}{current_lr:^9.5f}{self.lr:^9.5f}{monitor:^11s}{duration:^8.2f}"
            self.print_in_color(msg, (0, 67, 54), (55, 65, 80))

        if self.stop_count > self.stop_patience - 1:
            msg = f"Training halted at epoch {epoch + 1} after {self.stop_patience} adjustments of learning rate with no improvement."
            self.print_in_color(msg, (0, 255, 0), (55, 65, 80))
            self.model.stop_training = True

    def on_epoch_begin(self, epoch, logs=None):
        self.now = time.time()

    def print_in_color(self, text_msg, foreground_color, background_color):
        r_fg, g_fg, b_fg = foreground_color
        r_bg, g_bg, b_bg = background_color
        color_format = f"\33[38;2;{r_fg};{g_fg};{b_fg};48;2;{r_bg};{g_bg};{b_bg}m"
        print(color_format + text_msg)

        print("\33[0m")
