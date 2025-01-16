def preprocess_input(dataset, pixel_max_value):
    return [[pixel / pixel_max_value for pixel in image.convert("L").getdata()]
            for image in dataset['image']]

def preprocess_output(dataset, classes_number):
    return [[1.0 if i == int(label) else 0.0 for i in range(classes_number)] 
            for label in dataset['label']]