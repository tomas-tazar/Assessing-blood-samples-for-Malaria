import sys, os, cv2, time, argparse, configparser
sys.path.append('Segmentation')
sys.path.append('Classifier')
from evaluating import evaluated_image, evaluate_image_for_sequential_spatialattention, evaluate_image_for_vgg, display_labelled_image, vgg_display_labelled_image
from data_preprocessing import pre_processing_otsu, pre_processing_kmeans

config = configparser.ConfigParser()
if not config.read('config.ini'):
    config['DEFAULT'] = {'save_image': False,
                         'save_folder': False,
                         'save_information': False,
                         'n_eval_images': 10,
                         'number_of_clusters': 4,
                         'save_information_directory': 'Evaluated-Images-Info',
                         'save_directory': 'Evaluated-Images',
                         'default_folder_location': 'Dependencies/malaria/malaria-ucl/test/'
                         }
    with open ('config.ini', 'w') as configfile:
        config.write(configfile)
        
parser = argparse.ArgumentParser(description='Segment and evaluate images of malaria infected cells.', add_help=False)
parser.add_argument('--image', '--i', type=str, help='Path to the image to be segmented and evaluated.')
parser.add_argument('--folder', '--f', type=str, help='Path to the folder containing images to be segmented and evaluated.')
parser.add_argument('--default', '--d', action='store_true', help='Segment and evaluate the default images.')
parser.add_argument('--save', '--s', type=bool, help='Save the segmented and evaluated image to a specific output.')
parser.add_argument('--savefolder', '--sf', type=bool, help='Change the default directory where the segmented and evaluated images are saved')
parser.add_argument('--savefolderinfo', '--sfi', type=bool, help='Change the default directory where the text files with information about the segmented and evaluated images are saved.')
parser.add_argument('--saveinfo', '--si', type=bool, help='Text file with information about the segmented and evaluated image.')
parser.add_argument('--help', '--h', action='store_true', help='Displays the help menu.')
parser.add_argument('--none', nargs='?', const=True, default=True, help='Displays the default "malaria analyzer" intro message.')
args = parser.parse_args()

save_image = config['DEFAULT']['save_image']
save_folder = config['DEFAULT']['save_folder']
save_information = config['DEFAULT']['save_information']
n_eval_images = config['DEFAULT']['n_eval_images']
number_of_clusters = config['DEFAULT']['number_of_clusters']
save_information_directory = config['DEFAULT']['save_information_directory']
save_directory = config['DEFAULT']['save_directory']
default_folder_location = config['DEFAULT']['default_folder_location']

models = {
    "1": "Sequential Model",
    "2": "Sequential model with Spatial Attention",
    "3": "VGG16"
}

segmentation_methods = {
    "1": "Otsu's Thresholding",
    "2": "K-Means Clustering"
}

# ================================================================================================================= #

def user_image_segmentation(im, choice_segmentation, choice_model):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
    match choice_segmentation:
        case "1": # Otsu's thresholding
            regions_of_interest, contours, image_with_rois = pre_processing_otsu(im)        
            match choice_model:
                case "1":  
                    time_start = time.time()
                    predictions  = evaluated_image(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) 
                case "2":
                    time_start = time.time()
                    predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention
                case "3":
                    time_start = time.time()
                    vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                case _:
                    print("Invalid choice of model. Please try again.")
                    sys.exit(1)
                    
        case "2": # K-means clustering
            regions_of_interest, contours, image_with_rois = pre_processing_kmeans(im, int(number_of_clusters))     
            match choice_model:
                case "1":
                    time_start = time.time()
                    predictions  = evaluated_image(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) 
                case "2":
                    time_start = time.time()
                    predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention
                case "3":
                    time_start = time.time()
                    vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                    labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                case _:
                    print("Invalid choice of model. Please try again.")
                    sys.exit(1)
        case _:
            raise Exception("Invalid choice of segmentation method. Please try again.") 
               
    # -------------------- #
    
    final_time = time.time() - time_start
    return labelled_image, final_time, number_of_infected, number_of_uninfected, unk, total

# ================================================================================================================= #

def user_folder_segmentation(directory, choice_segmentation, choice_model):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    result_array = []
    
    n_inf = 0
    n_unf = 0
    n_unk = 0
    t = 0

    # Otsu's thresholding
    match choice_segmentation:
        case "1": # Otsu's thresholding
            match choice_model:
                case "1":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)        
                                predictions = evaluated_image(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) # Seuqential
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case "2":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)        
                                predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case "3":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)        
                                vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case _:
                    print("Invalid choice of model, please try again.")
                    sys.exit(1)
                    
        case "2": # K-means
            match choice_model:
                case "1":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))        
                                predictions = evaluated_image(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) # Seuqential
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case "2":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))     
                                predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case "3":
                    time_start = time.time()
                    for file in os.listdir(directory):
                        if (file.endswith(".jpg")) or (file.endswith(".png")) or (file.endswith(".jpeg")) or (file.endswith(".tif")) or (file.endswith(".tiff")):
                            try:
                                # file_path = directory + file # Linux compatibility
                                file_path = directory + '/' + file # Windows compatibility
                                regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))        
                                vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                                labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                                n_inf += number_of_infected
                                n_unf += number_of_uninfected
                                n_unk += unk
                                t += total
                                result_array += [labelled_image]
                            except:
                                raise FileNotFoundError("Could not find the file: ", file_path)
                case _:
                    print("Invalid choice of model, please try again.")
                    sys.exit(1)
        case _:
            raise Exception("Invalid choice of segmentation method. Please try again.")
        
    # -------------------- # 
    
    final_time = time.time() - time_start
    return result_array, final_time, n_inf, n_unf, n_unk, t

# ================================================================================================================= #

def default_folder_segmentation(n_images, choice_segmentation, choice_model):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    ucl_test_images = []
    result_array = []
    file_extenstion = '.jpg'

    n_inf = 0
    n_unf = 0
    n_unk = 0
    t = 0

    for file in os.listdir(default_folder_location):
        if file.endswith(file_extenstion): ucl_test_images.append(file)
        
    match choice_segmentation:
        case "1": # Otsu's thresholding
            match choice_model:
                case "1": 
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)
                            predictions = evaluated_image(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) # Sequential 
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path)  
                case "2":
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)
                            predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention    
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path) 
                case "3":
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_otsu(file_path)
                            vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path)
                
                case _: 
                    print("Invalid choice. Please try again.")
                    sys.exit(1)
                    
        case "2": # K-Means clustering
            match choice_model:
                case "1": 
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))
                            predictions = evaluated_image(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions, image_with_rois) # Sequential 
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path)  
                case "2":
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))
                            predictions_spatialattention = evaluate_image_for_sequential_spatialattention(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = display_labelled_image(contours, predictions_spatialattention, image_with_rois) # Sequential with Spatial Attention    
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path) 
                case "3":
                    time_start = time.time()
                    for file in ucl_test_images[:n_images]:
                        try:
                            file_path = default_folder_location + file
                            regions_of_interest, contours, image_with_rois = pre_processing_kmeans(file_path, int(number_of_clusters))
                            vgg_predictions = evaluate_image_for_vgg(regions_of_interest)
                            labelled_image, number_of_infected, number_of_uninfected, unk, total = vgg_display_labelled_image(contours, vgg_predictions, image_with_rois) # VGG16
                            n_inf += number_of_infected
                            n_unf += number_of_uninfected
                            n_unk += unk
                            t += total
                            result_array += [labelled_image]
                        except:
                            raise FileNotFoundError("Could not find the file: ", file_path)
                
                case _: 
                    print("Invalid choice of model. Please try again.")
                    sys.exit(1)
        case _:
            raise Exception("Invalid choice of segmentation method. Please try again.")
    # -------------------- #         
        
    final_time = time.time() - time_start
    return result_array, final_time, n_inf, n_unf, n_unk, t

# ================================================================================================================= #

def main(): # Main function for the program
    global save_image
    global save_folder
    global save_information
    global save_information_directory
    global save_directory
    
    # ================================================================================================================= #

    if args.savefolder:
        save_folder = True
        with open('config.ini', 'w') as configfile:
            config.set('DEFAULT', 'save_folder', 'True')
            config.set('DEFAULT', 'save_directory', sys.argv[2])
            config.write(configfile)
        print("\nSave folder set to: ", sys.argv[2])
        
    elif args.savefolderinfo:
        save_information = True
        with open('config.ini', 'w') as configfile:
            config.set('DEFAULT', 'save_information', 'True')
            config.set('DEFAULT', 'save_information_directory', sys.argv[2])
            config.write(configfile)
        print("\nSave folder for information set to: ", sys.argv[2])
        
    # ================================================================================================================= #
    
    elif args.image: # Single image segmentation and evaluation
        if os.path.isfile(sys.argv[2]):
            print('\nWhich of the following segmentation methods do you want to use? (1/2) ')
            print({key: value for key, value in segmentation_methods.items()})  # Use dictionary comprehension to print the segmentation methods
            choice_segmentation = input(">>> ")
            print("\nWhich of the following models do you want to use for evaluation? (1/2/3) ")
            print({key: value for key, value in models.items()})  # Use dictionary comprehension to print the models
            choice_models = input(">>> ")
            print(f"\nSegmenting and evaluating image '{sys.argv[2]}', this might take a while... ")
            segmented_im, time, number_of_infected, number_of_uninfected, unknown, total = user_image_segmentation(sys.argv[2], choice_segmentation, choice_models)
            if segmented_im is None or segmented_im.size == 0: # Image is not read correctly
                print("\nCould not find the image, please try again with a valid image.")
                sys.exit()
            
            print("+-----------------------------------------------------------------+")
            print("\nSegmentation and evaluation took: ", time, " seconds", "\nPress 'q' to quit...", "\n")
            print("Number of infected cells: ", number_of_infected)
            print("Number of uninfected cells: ", number_of_uninfected)
            print("Number of unknown cells: ", unknown)
            print("Percentage of infected cells in the image: ", number_of_infected/total * 100,"%" + "\n")
            print("+-----------------------------------------------------------------+")
            
            if args.saveinfo:
                try:
                    save_information = True
                    curr_save_information_directory = save_information_directory
                    if not os.path.isdir(curr_save_information_directory):
                        os.mkdir(curr_save_information_directory)
                        print(f"Created directory '{curr_save_information_directory}' for saving the information about the segmented and evaluated images.")
                    save_file = curr_save_information_directory + '/' + sys.argv[4] # Text file name
                    print(f"Saving the information about the segmented and evaluated images to '{save_file}'.")
                    if os.path.isfile(save_file):
                        print(f"\nFile '{save_file}' already exists! Do you want to overwrite it? (y/n)")
                        overwrite = input('>>> ')
                        if overwrite == 'y':
                            print("Overwriting file...")
                            pass
                        else:
                            print("\nNo changes made, considering naming the output file something else...")
                            sys.exit(0)
                    with open(save_file, 'w') as f:
                        f.write("+-----------------------------------------------------------------+")
                        f.write(f"\nImage name: {sys.argv[2]}\n")
                        f.write(f"Segmentation method used: {segmentation_methods[choice_segmentation]}\n")
                        f.write(f"Model used: {models[choice_models]}\n")
                        f.write(f"Segmentation and evaluation took: {time} seconds\n")
                        f.write(f"Number of infected cells: {number_of_infected}\n")
                        f.write(f"Number of uninfected cells: {number_of_uninfected}\n")
                        f.write(f"Number of unknown cells: {unknown}\n")
                        f.write(f"Percentage of infected cells in the image: {number_of_infected/total * 100}%\n")
                        f.write("+-----------------------------------------------------------------+")
                except:
                    print(f"\nCould not save the information about the segmented and evaluated image, please check for valid output file name.")
                    sys.exit(1)
            elif args.save:
                try:
                    save_image = True
                    curr_save_directory = save_directory
                    if not os.path.isdir(curr_save_directory):
                        os.mkdir(curr_save_directory)
                        print(f"Created directory '{curr_save_directory}'")
                    save_directory_image = curr_save_directory + '/' + sys.argv[4]
                    print(f"Saving segmented and evaluated image to '{save_directory_image}'")
                    if os.path.isfile(save_directory_image):
                        print(f"\nFile '{save_directory_image}' already exists in the directory! Do you want to overwrite it? (y/n)")
                        overwrite = input('>>> ')
                        if overwrite == 'y':
                            print("Overwriting file...")
                            pass
                        else:
                            print("\nNo changes made, considering naming the output file something else...")
                            sys.exit(0)
                    cv2.imwrite(save_directory_image, segmented_im)
                except:
                    print("Could not save the image, please check for valid saving path or valid image name.")
                    sys.exit(1)
            else:
                sys.stdout = open(os.devnull, 'w')
                cv2.imshow("Segmented and evaluated image", segmented_im)
                while True:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                sys.stdout = sys.__stdout__
        else:
            print("\nCould not find the image, please check argument or try again with a valid image.")
            sys.exit(1)
    
    # ================================================================================================================= #

    elif args.folder: # Folder segmentation and evaluation
        if os.path.isdir(sys.argv[2]):
            if len(os.listdir(sys.argv[2])) == 0:
                print(f"\nFolder {sys.argv[2]} is empty, please try again with a folder containing images.")
                sys.exit(1)
            
            print('\nWhich of the following segmentation methods do you want to use? (1/2) ')
            print({key: value for key, value in segmentation_methods.items()})  # Use dictionary comprehension to print the segmentation methods
            choice_segmentation = input(">>> ")
            print("\nWhich of the following models do you want to use for evaluation? (1/2/3) ")
            print({key: value for key, value in models.items()})  # Use dictionary comprehension to print the models
            choice_models = input(">>> ")
            print(f"\nSegmenting and evaluating images in folder '{sys.argv[2]}', this might take a while..." , "\n")
            user_segmented_folder, time, number_of_infected, number_of_uninfected, unknown, total  = user_folder_segmentation(sys.argv[2], choice_segmentation, choice_models)
            
            print("+-----------------------------------------------------------------+\n")
            print(len(user_segmented_folder), " images segmented and evaluated.")
            print("Segmentation and evaluation took: ", time, " seconds", "\n")
            print("Number of infected cells: ", number_of_infected)
            print("Number of uninfected cells: ", number_of_uninfected)
            print("Number of unknown cells: ", unknown)
            print("Percentage of infected cells in the images: ", number_of_infected/total * 100,"%", "\n")
            print("+-----------------------------------------------------------------+")
            
            if args.saveinfo:
                try:
                    save_information = True
                    curr_save_information_directory = save_information_directory
                    if not os.path.isdir(curr_save_information_directory):
                        os.mkdir(curr_save_information_directory)
                        print(f"Created directory '{curr_save_information_directory}' for saving the information about the segmented and evaluated images.")
                    save_file = curr_save_information_directory + '/' + sys.argv[4] # Text file name
                    print(f"Saving the information about our evaluated folder to '{save_file}'.")
                    if os.path.isfile(save_file):
                        print(f"\nFile '{save_file}' already exists! Do you want to overwrite it? (y/n)")
                        overwrite = input('>>> ')
                        if overwrite == 'y':
                            print("Overwriting file...")
                            pass
                        else:
                            print("\nNo changes made, considering naming the output file something else...")
                            sys.exit(0)
                    with open(save_file, 'w') as f:
                        f.write("+-----------------------------------------------------------------+")
                        f.write(f"\nFolder name: {sys.argv[2]}\n")
                        f.write(f"Segmentation method used: {segmentation_methods[choice_segmentation]}\n")
                        f.write(f"Model used: {models[choice_models]}\n")
                        f.write(f"Segmentation and evaluation took: {time} seconds\n")
                        f.write(f"Number of infected cells: {number_of_infected}\n")
                        f.write(f"Number of uninfected cells: {number_of_uninfected}\n")
                        f.write(f"Number of unknown cells: {unknown}\n")
                        f.write(f"Percentage of infected cells in the folder: {number_of_infected/total * 100}%\n")
                        f.write("+-----------------------------------------------------------------+")
                except:
                    print(f"\nCould not save the information about the segmented and evaluated folder, please check for valid output file name.")
                    sys.exit(1)
                    
            elif args.save:
                idx = 0
                try:
                    while True:
                        save_image = True
                        curr_save_directory = save_directory
                        
                        if not os.path.isdir(curr_save_directory):
                            os.mkdir(curr_save_directory)
                            print(f"Created directory '{curr_save_directory}'")
                        save_directory_image = curr_save_directory + '/' + sys.argv[4]
                        if not os.path.isdir(save_directory_image):
                            os.mkdir(save_directory_image)
                        image_attachment = ('evaluated-image-' + str(idx) +'.jpg')
                        if os.path.isfile(save_directory_image + image_attachment):
                            print(f"\nFile '{save_directory_image + image_attachment}' already exists in the directory! Do you want to overwrite it? (y/n)")
                            overwrite = input('>>> ')
                            if overwrite == 'y':
                                print("Overwriting file...")
                                pass
                            else:
                                print("\nNo changes made, considering naming the output file something else...")
                                sys.exit(0)
                        final_dir = save_directory_image + image_attachment
                        cv2.imwrite(final_dir, user_segmented_folder[idx])
                        idx += 1
                        if idx == len(user_segmented_folder):
                            print(f"Saved evaluated images to folder: '{save_directory_image}'")
                            break
                except:
                    print("Could not save the image, please check for valid saving path or valid image name.")
                    sys.exit(1)
            else:
                print("\nPress 'q' to quit the view, 'b' to go back to the previous image, 'n' to go forward to the next image.", "\n")
                sys.stdout = open(os.devnull, 'w')
                idx = 0
                cv2.imshow("Segmented and evaluated image", user_segmented_folder[0])
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    elif key == ord('b'):
                        idx = (idx - 1) % len(user_segmented_folder)
                        cv2.imshow("Segmented and evaluated image", user_segmented_folder[idx])
                    elif key == ord('n'):
                        idx = (idx + 1) % len(user_segmented_folder)
                        cv2.imshow("Segmented and evaluated image", user_segmented_folder[idx])
                sys.stdout = sys.__stdout__
        else:
            print("\nCould not find the folder, please check the argument or try again with a valid folder.")
            sys.exit(1)
        
    # ================================================================================================================= #

    elif args.default: # Default folder evaluation        
        print('\nWhich of the following segmentation methods do you want to use? (1/2) ')
        print({key: value for key, value in segmentation_methods.items()})  # Use dictionary comprehension to print the segmentation methods
        choice_segmentation = input(">>> ")
        print("\nWhich of the following models do you want to use for evaluation? (1/2/3) ")
        print({key: value for key, value in models.items()})  # Use dictionary comprehension to print the models
        choice_models = input(">>> ")
        print(f"\nSegmenting and evaluating default folder of images 'Dependencies/malaria/malaria-ucl/test/', this might take a while...")
        segmented_folder, time, number_of_infected, number_of_uninfected, unknown, total = default_folder_segmentation(int(n_eval_images), choice_segmentation, choice_models)
        
        print("+-----------------------------------------------------------------+\n")
        print(len(segmented_folder), " images segmented and evaluated.", "\n")
        print("Segmentation and evaluation took: ", time, " seconds", "\n")
        print("Number of infected cells: ", number_of_infected)
        print("Number of uninfected cells: ", number_of_uninfected)
        print("Number of unknown cells: ", unknown)
        print("Percentage of infected cells in the images: ", number_of_infected/total * 100,"%", "\n")
        print("+-----------------------------------------------------------------+")
        
        if args.saveinfo:
            try:
                save_information = True
                curr_save_information_directory = save_information_directory
                if not os.path.isdir(curr_save_information_directory):
                    os.mkdir(curr_save_information_directory)
                    print(f"Created directory '{curr_save_information_directory}' for saving the information about the segmented and evaluated images.")
                save_file = curr_save_information_directory + '/' + sys.argv[3] # Text file name
                print(f"Saving the information about the default folder to '{save_file}'.")
                if os.path.isfile(save_file):
                    print(f"\nFile '{save_file}' already exists! Do you want to overwrite it? (y/n)")
                    overwrite = input('>>> ')
                    if overwrite == 'y':
                        print("Overwriting file...")
                        pass
                    else:
                        print("\nNo changes made, considering naming the output file something else...")
                        sys.exit(0)
                with open(save_file, 'w') as f:
                    f.write("+-----------------------------------------------------------------+")
                    f.write(f"\nFolder name: {'Default UCL-test dataset'}\n")
                    f.write(f"Segmentation method used: {segmentation_methods[choice_segmentation]}\n")
                    f.write(f"Model used: {models[choice_models]}\n")                    
                    f.write(f"Segmentation and evaluation took: {time} seconds\n")
                    f.write(f"Number of infected cells: {number_of_infected}\n")
                    f.write(f"Number of uninfected cells: {number_of_uninfected}\n")
                    f.write(f"Number of unknown cells: {unknown}\n")
                    f.write(f"Percentage of infected cells in the folder: {number_of_infected/total * 100}%\n")
                    f.write("+-----------------------------------------------------------------+")
            except:
                print(f"\nCould not save the information the default folder, please check for valid output file name.")
                sys.exit(1)
                    
        elif args.save:
            idx = 0
            try:
                while True:
                    save_image = True
                    curr_save_directory = save_directory
                    
                    if not os.path.isdir(curr_save_directory):
                        os.mkdir(curr_save_directory)
                        print(f"Created directory '{curr_save_directory}'")
                    save_directory_image = curr_save_directory + '/' + sys.argv[3]
                    if not os.path.isdir(save_directory_image):
                        os.mkdir(save_directory_image)
                    image_attachment = ('evaluated-image-' + str(idx) +'.jpg')
                    if os.path.isfile(save_directory_image + image_attachment):
                        print(f"\nFile '{save_directory_image + image_attachment}' already exists in the directory! Do you want to overwrite it? (y/n)")
                        overwrite = input('>>> ')
                        if overwrite == 'y':
                            print("Overwriting file...")
                            pass
                        else:
                            print("\nNo changes made, considering naming the output file something else...")
                            sys.exit(0)
                    final_dir = save_directory_image + image_attachment
                    cv2.imwrite(final_dir, segmented_folder[idx])
                    idx += 1
                    if idx == len(segmented_folder):
                        print(f"Saved evaluated images to folder: '{save_directory_image}'")
                        break
            except:
                print("Could not save the image, please check for valid saving path or valid image name.")
                sys.exit(1)
        else:
            print("\nPress 'q' to quit the view, 'b' to go back to the previous image, 'n' to go forward to the next image.")
            try:
                sys.stdout = open(os.devnull, 'w')
                idx = 0
                cv2.imshow("Segmented and evaluated image", segmented_folder[0])
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    elif key == ord('b'):
                        idx = (idx - 1) % len(segmented_folder)
                        cv2.imshow("Segmented and evaluated image" , segmented_folder[idx])
                    elif key == ord('n'):
                        idx = (idx + 1) % len(segmented_folder)
                        cv2.imshow("Segmented and evaluated image", segmented_folder[idx])
                sys.stdout = sys.__stdout__
            except:
                print("\nSomething has gone wrong, images from default folder could not be displayed.")
                sys.exit(1)    
                
    # ================================================================================================================= #
    
    elif args.save: # Save the segmented images in a folder
        print("Saving enabled...")
        
    # ================================================================================================================= #

    elif args.help: # More advanced help menu with examples
        print(''' 
    * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - *

            1. To remedy some basic errors, make sure you have done the following: 
                - Include the image you want to segment and evaluate in the same folder as the program.
                - Include the folder containing the images you want to segment and evaluate in the same folder as the program.

            2. Below are some example command line inputs to the program:
                - "python3 malaria_segmentation_and_evaluation.py --image {image.jpg}" -> This displays the segmented and evaluated image.
                - "python3 malaria_segmentation_and_evaluation.py --folder {folder/}" -> This displays the segmented and evaluated images in the folder.
                - "python3 malaria_segmentation_and_evaluation.py --i {image.jpg} --save {output.jpg}" -> This saves the evaluated image to an output image.
                - "python3 malaria_segmentation_and_evaluation.py --f {folder/} --save {outputfolder/}" -> This saves the evaluated folder to an output folder.
                - "python3 malaria_segmentation_and_evaluation.py --d --s {outputfolder/}" -> This saves the evaluated default folder to an output folder.
                - "python3 malaria_segmentation_and_evaluation.py --i {image.jpg} --saveinfo {output.txt}" -> This saves the evaluation information to an output file.
                - "python3 malaria_segmentation_and_evaluation.py --f {folder/} --saveinfo {output.txt}" -> This saves the folders evaluation information to an output file.
                - "python3 malaria_segmentation_and_evaluation.py --d --si {output.txt}" -> This saves the default folders evaluation information to an output file.
                - "python3 malaria_segmentation_and_evaluation.py --sf {outputfolder/}" -> This changes the default folder to the output folder the user specifies.
                - "python3 malaria_segmentation_and_evaluation.py --sfi {outputfolder/}" -> This changes the default information folder to the output folder the user specifies.

            3. Make sure to have the following libraries installed before running the program: 
                - numpy, opencv-python, tensorflow-cpu, matplotlib.
                - and are using python 3.10 or higher.
                
            4. The user can edit the config file to change the, default output folder, the default information folder and other characteristics of the program.
            
            When displaying folders of images, the user can press 'n' to to go to the next image, 'b' to go back to the previous image, and 'q' to quit the view.

    * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - *
            ''')

    # ================================================================================================================= #

    elif args.none: # If no arguments are given, print the help message. 
        print('''
    * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  *       
       __    __     ______     __         ______     ______     __     ______        ______     __   __     ______     __         __  __     ______     ______     ______    
      /\ "-./  \   /\  __ \   /\ \       /\  __ \   /\  == \   /\ \   /\  __ \      /\  __ \   /\ "-.\ \   /\  __ \   /\ \       /\ \_\ \   /\___  \   /\  ___\   /\  == \   
      \ \ \-./\ \  \ \  __ \  \ \ \____  \ \  __ \  \ \  __<   \ \ \  \ \  __ \     \ \  __ \  \ \ \-.  \  \ \  __ \  \ \ \____  \ \____ \  \/_/  /__  \ \  __\   \ \  __<   
       \ \_\ \ \_\  \ \_\ \_\  \ \_____\  \ \_\ \_\  \ \_\ \_\  \ \_\  \ \_\ \_\     \ \_\ \_\  \ \_\\"\_\  \ \_\ \_\  \ \_____\  \/\_____\   /\_____\  \ \_____\  \ \_\ \_\ 
        \/_/  \/_/   \/_/\/_/   \/_____/   \/_/\/_/   \/_/ /_/   \/_/   \/_/\/_/      \/_/\/_/   \/_/ \/_/   \/_/\/_/   \/_____/   \/_____/   \/_____/   \/_____/   \/_/ /_/ 

    * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  *       

           This program is designed to segment, evaluate and display images of cells infected with malaria, it is adapted to be intuitive and interactive and works as follows:
           
        * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  *
           
            1. Display an image by providing the name of the image after the '--image' or '--i' argument. The images can be a .jpg, .png or .tif file and should be the 
               first thing after the argument in the command line.
                
            2. Display a folder of images by providing the folder name after the '--folder' or '--f' argument. The folder should contain images of cells that are to be
               evaluated. The folder should contain .jpgs, .pngs or .tif files and should be provided first thing after the argument in the command line.
            
            3. By providing the argument '--default' or '--d'. In this case the program will segment and evaluate a 'UCL-test' default folder of images. 
            
            4. The user has the option to save the evaluated images by providing the argument '--save' or '--s' after the name of the image and then specifying the output
               image. Likewise, the user can save an evaluated folder by providing the argument '--save' or '--s' after the name of the folder to be evaluated and 
               then specifying the output folder. Please be careful to also include the file extension when specifying the output image (e.g. .jpg) and the output folder
               (exampleFolder/).
                
               By default the program will save the evaluated folders/images to the folder 'Evaluated-Images' in the same directory as the program. If the folder does not
               exist, the program will create it.
                                    
            5. It is also possible to save information about the evaluation by providing the argument '--saveinfo' or '--si'. This functions the same as the '--s' argument.
               The output file should be specified to be a .txt file, and will be saved to the folder 'Evaluated-Image-Info' in the same directory as the program.
               
            6. The user can change the default save folder by providing the argument '--savefolder' or '--sf' and then specifying the name of the folder to be used.
            
            7. The user can change the default information folder by providing the argument '--savefolderinfo' or '--sfi' and then specifying the name of the folder to be used.
            
            8. By providing the argument '--help' or '--h' to the program. In this case the program will output a more detaield guide to help the user use the program.
            
            9. By running the program without any arguments. In this case the program will output this message to guide the user on how to use the program.
            
        * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  *

    * -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  *       
        ''')
        sys.exit(1)
    
if __name__ == "__main__":
    main()