RECOGNITION_DATASET=timit
RECOGNITION_PREPROCESS_TRAIN_SOURCE=data/$(RECOGNITION_DATASET)/train
RECOGNITION_PREPROCESS_TEST_SOURCE=data/$(RECOGNITION_DATASET)/test
RECOGNITION_PREPROCESS_TRAIN_TARGET=data/recognition/train.pkl
RECOGNITION_PREPROCESS_TEST_TARGET=data/recognition/test.pkl # comment starting from the "=" if you don't want to use a testing set
RECOGNITION_MODEL_TARGET=data/recognition/model.pt

.PHONY: recognition_test recognition_train clean

clean:
	rm -rf $(RECOGNITION_PREPROCESS_TRAIN_TARGET) $(RECOGNITION_PREPROCESS_TEST_TARGET) $(RECOGNITION_MODEL_TARGET)

$(RECOGNITION_PREPROCESS_TRAIN_TARGET): $(RECOGNITION_PREPROCESS_TRAIN_SOURCE)
	python -m preprocess.preprocess_$(RECOGNITION_DATASET) $(RECOGNITION_PREPROCESS_TRAIN_SOURCE) $(RECOGNITION_PREPROCESS_TRAIN_TARGET)

$(RECOGNITION_PREPROCESS_TEST_TARGET): $(RECOGNITION_PREPROCESS_TEST_SOURCE)
	python -m preprocess.preprocess_$(RECOGNITION_DATASET) $(RECOGNITION_PREPROCESS_TEST_SOURCE) $(RECOGNITION_PREPROCESS_TEST_TARGET)

preprocess_recognition: $(RECOGNITION_PREPROCESS_TRAIN_TARGET)

$(RECOGNITION_MODEL_TARGET): $(RECOGNITION_PREPROCESS_TRAIN_TARGET) $(RECOGNITION_PREPROCESS_TEST_TARGET)
	python -m train.recognition_$(RECOGNITION_DATASET) $(RECOGNITION_PREPROCESS_TRAIN_TARGET) $(RECOGNITION_MODEL_TARGET) $(RECOGNITION_PREPROCESS_TEST_TARGET)

train_recognition: $(RECOGNITION_MODEL_TARGET)
