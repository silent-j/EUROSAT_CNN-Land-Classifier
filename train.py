import argparse
import numpy as np

from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from utils import load_generators
from utils import compile_model, plot_history
from utils import plot_predictions, display_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', action='store', type=str)
    parser.add_argument('test_dir', action='store', type=str)
    parser.add_argument('save_path', action='store', type=str)
    parser.add_argument('--epochs', action='store', dest='epochs',
                       type=int, default=10)
    parser.add_argument('--lr', action='store', dest='learn_rate',
                       type=float, default=0.01)
    parser.add_argument('--batch_size', action='store', dest='batch_size',
                       type=int, default=64)
    parser.add_argument('--fine_tune', action='store_true',
                        default=True)
    parser.add_argument('--eval', action='store_true',
                       default=True)  
    return parser.parse_args()


if __name__=="__main__":

    args = parse_args()
    TRAIN_DIR  = args.train_dir
    TEST_DIR = args.test_dir
    SAVE_PATH = args.save_path
    N_EPOCHS = args.epochs
    NUM_CLASSES = 10
    BATCH_SIZE = args.batch_size 
    INPUT_SHAPE = (64, 64, 3)
    FINE_TUNE = args.fine_tune
    EVAL = args.eval
    
    if FINE_TUNE:
        print("fine-tuning is on...")
    # load image generators
    train_gen, valid_gen = load_generators(TRAIN_DIR, batch_size=BATCH_SIZE, val_split=0.3)
    
    N_STEPS = train_gen.samples//BATCH_SIZE
    N_VAL_STEPS = valid_gen.samples//BATCH_SIZE
    
    # optimizer
    optim = Adagrad(lr=args.learn_rate)
    
    # compile the model
    print("compiling..")
    model = compile_model(INPUT_SHAPE, NUM_CLASSES, optim, fine_tune=None)
    model.summary()
    
    print("loading model callbacks..")
    MODEL_WEIGHTS = '\\'.join(SAVE_PATH.split('\\')[:-1])+'\\model.weights.best.hdf5' # use model's save path to also save weights
    checkpoint = ModelCheckpoint(filepath=MODEL_WEIGHTS, #save weights in same loc as model
                        monitor='val_categorical_accuracy',
                        save_best_only=True,
                        verbose=1)

    early_stop = EarlyStopping(monitor='val_categorical_accuracy',
                               patience=10,
                               restore_best_weights=True,
                               mode='max')
    
    print("training model..")
    history = model.fit_generator(train_gen,
                                 steps_per_epoch=N_STEPS,
                                 epochs=N_EPOCHS,
                                 callbacks=[early_stop, checkpoint],
                                 validation_data=valid_gen,
                                 validation_steps=N_VAL_STEPS)
            
    # fine-tuning is on by default
    if FINE_TUNE:
        
        train_gen.reset()
        valid_gen.reset()
        model = compile_model(INPUT_SHAPE, NUM_CLASSES, optim, fine_tune=14)
        
        print("fine-tuning model..")
        history = model.fit_generator(train_gen,
                                     steps_per_epoch=N_STEPS,
                                     epochs=N_EPOCHS,
                                     callbacks=[early_stop, checkpoint],
                                     validation_data=valid_gen,
                                     validation_steps=N_VAL_STEPS)        
    plot_history(history)
        
    if EVAL:        
        print("evaluating predictions..")
            
        model.load_weights(MODEL_WEIGHTS)
        
        test_gen = ImageDataGenerator(rescale=1./255.0)
        test_gen = test_gen.flow_from_directory(
                        directory=TEST_DIR, target_size=(64, 64),
                        batch_size=1, class_mode=None,
                        color_mode='rgb', shuffle=False,
                        seed=69)
        
        class_indices = train_gen.class_indices
        class_indices = dict((v,k) for k,v in class_indices.items())
        test_gen.reset()

        predictions = model.predict_generator(test_gen, steps=len(test_gen.filenames))
        predicted_classes = np.argmax(np.rint(predictions), axis=1)
        true_classes = test_gen.classes

        prf, conf_mat = display_results(true_classes, predicted_classes, class_indices.values())
        
        print(prf)
        
        plot_predictions(true_classes, predictions, test_gen, class_indices)
    
    print("saving model..")
    model.save(SAVE_PATH)
