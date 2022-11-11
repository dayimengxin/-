from utils import *
from tqdm import tqdm
from paddle import inference
import os, cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import pickle
from matplotlib import pyplot as plt
import glob
IMAGE_TYPE = '.jpg'


class FaceEval:
    def __init__(self):
        self.threshold = 0.4
        self.mtcnn = MTCNN()
        self.face_eval = self.init_resnet50_predictor('../model/Backbone')
        self.face_db_path = 'FaceDatabase'
        self.face_data_path = 'face_data.fdb'
        self.face_db = self.load_face_data()
        self.mtcnn_input_scale = 0.4  # 缩放图片加快计算

    def update_face_data(self):
        '''
        用于更新人脸数据库
        :return:
        '''
        face_db = {}
        assert os.path.exists(self.face_db_path), 'face_db_path {} not exist'.format(self.face_db_path)
        for path in tqdm(os.listdir(self.face_db_path)):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(self.face_db_path, path)
            # print(image_path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img, img)
            if imgs is None or len(imgs) > 1:
                print('人脸库中的 %s 图片包含不是1张人脸，自动跳过该图片' % image_path)
                continue
            imgs = self.process(imgs)
            feature = self.infer(imgs)
            face_db[name] = feature[0]
        with open(self.face_data_path, "wb") as f:
            pickle.dump(face_db, f)
        print('finished faceDatabase transform!')
        return face_db

    def load_face_data(self):
        if not os.path.exists(self.face_data_path):
            print('face_data_path not exist!,try to get faceDatabase transform!')
            face_db = self.update_face_data()
            return face_db
        with open(self.face_data_path, "rb") as f:
            face_db = pickle.load(f)
        print('finished load face_data!')
        return face_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        if len(imgs1) > 1:
            imgs = np.array(imgs1).astype('float32')
        else:
            imgs = imgs1[0][np.newaxis, :].astype('float32')
        return imgs

    @staticmethod
    def init_resnet50_predictor(model_dir):
        model_file = model_dir + '.pdmodel'
        params_file = model_dir + '.pdiparams'
        config = inference.Config()
        config.set_prog_file(model_file)
        config.set_params_file(params_file)
        config.use_gpu()
        config.enable_use_gpu(500, 0)
        predictor = inference.create_predictor(config)
        return predictor

    def infer(self, imgs):
        '''
        人脸对比
        :param img:
        :return:
        '''

        # 获取输入的名称
        input_names = self.face_eval.get_input_names()
        handle_image = self.face_eval.get_input_handle(input_names[0])
        # 设置输入
        input_img_size = imgs.shape
        handle_image.reshape([input_img_size[0], 3, input_img_size[2], input_img_size[3]])
        handle_image.copy_from_cpu(imgs)
        # 运行predictor
        self.face_eval.run()
        # 获取输出
        output_names = self.face_eval.get_output_names()
        features = self.face_eval.get_output_handle(output_names[0])
        features = features.copy_to_cpu()  # numpy.ndarray类型
        #print (features[0])
        return features

    def recognition(self, img):
        orimg_shape = img.shape
        resize_img = cv2.resize(img, (int(orimg_shape[1] * self.mtcnn_input_scale), int(orimg_shape[0] * self.mtcnn_input_scale)))
        imgs, boxes = self.mtcnn.infer_image(resize_img, img, self.mtcnn_input_scale)
        if imgs is None:
            return None, None
        imgs = self.process(imgs)
        features = self.infer(imgs)
        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i]
            results_dict = {}
            for name in self.face_db.keys():
                feature1 = self.face_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
            else:
                names.append('unknow')
        return boxes, names

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simsun.ttc', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, img, boxes_c, names):
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 2)
                # 判别为人脸的名字
                # font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                # img = cv2.putText(img, name, (corpbbox[0], corpbbox[1]), font, 0.5, (0, 255, 0), 1)
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] + 25, color=(255, 255, 0), size=30)
        cv2.imshow("result", img)
        cv2.waitKey(1)
        return img

######### New for this project #########

class WCFaceEval(FaceEval):

    def __init__(self, data_root, model_root, probe_list, gallery_list):
        self.threshold = 0.3
        #self.mtcnn = MTCNN()
        self.mtcnn = None
        self.data_root = data_root
        self.face_eval = self.init_resnet50_predictor(model_root)
        self.probe_list = probe_list
        self.gallery_list = gallery_list
        self.gallery_data_path = 'gallery_data.fdb'
        self.gallery_db = self.load_gallery_data()
        self.mtcnn_input_scale = 1.0  # 缩放图片input_list加快计算 0.4

    def load_list_file(self, input_list):
        data = list()
        with open(input_list, 'r') as file:
            all_lines = file.readlines()
            for line in all_lines:
                image = line[-7:].rstrip()
                label = line[:-7].rstrip()
                data.append([label, image])
        return data
    

    def update_gallery_data(self):
        '''
        用于更新 Gallery 人脸数据库
        :return:
        '''
        face_db = {}
        assert os.path.isfile(self.gallery_list), 'gallery data list file {} not exist'.format(self.gallery_list)
        
        gallery_data = self.load_list_file(self.gallery_list)
       
        for i in tqdm(range(len(gallery_data)), ncols=80):
            label = gallery_data[i][0]
            image_name = gallery_data[i][1]
            name = '%s+%s'%(label, image_name)
            
            gallery_images = glob.glob(os.path.join(self.data_root, label, 'P*.jpg'))
            if image_name[0] == 'C':
                gallery_images = glob.glob(os.path.join(self.data_root, label, 'C*.jpg'))
            for image_path in gallery_images:

                image_name = os.path.basename(image_path)[:-4]
                name = '%s+%s'%(label, image_name)
                #print ('face db key: ', name)
                #image_path = os.path.join(self.data_root, label, image_name + IMAGE_TYPE) 
                #print('Loading gallery image: %s ...' % image_path)
                
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                imgs = [img]
            
                imgs = self.process(imgs)
                feature = self.infer(imgs)
                face_db[name]= feature[0]

        with open(self.gallery_data_path, "wb") as f:
            pickle.dump(face_db, f)
        print('finished faceDatabase transform!')

        return face_db
        

    def load_gallery_data(self):
        if not os.path.exists(self.gallery_data_path):
            print('face_data_path not exist!,try to get faceDatabase transform!')
            face_db = self.update_gallery_data()
            return face_db
        with open(self.gallery_data_path, "rb") as f:
            face_db = pickle.load(f)
        print('finished load face_data!')
        return face_db
    
    def recognition(self, img):
    
        imgs = [img]
        imgs = self.process(imgs)
        features = self.infer(imgs)

        #print('len of features', len(features))

        names = []
        probs = []
        for i in range(len(features)):
            feature = features[i]
            #print ('feature: ', feature)
            results_dict = {}
            for name in self.gallery_db.keys():
                #print ('name', name)
                feature1 = self.gallery_db[name]
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob

                #diff = np.subtract(feature, feature1)
                #dist = np.sum(np.square(diff))
                #results_dict[name] = dist
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            #results = sorted(results_dict.items(), key=lambda d: d[1])


            #names = self.voting(results)
            #print ('res', results[:10])
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append((name, prob))
            #else:
                #names.append(('unknow', prob))
        return names

    def voting(self, results):
        top = 10
        vote = dict()
        
        for res in results[:top]:
            name = res[0].split('+')[0]
            prob = res[1]

            if name in vote:
                vote[name][0] += 1
                if prob > vote[name][2]:
                    vote[name][1] = res[0]
                    vote[name][2] = prob
            else:
                vote[name] = [1, res[0], prob]

        results = sorted(vote.items(), key=lambda d: (d[1][0], d[1][2]), reverse=True)
        #print (results)
        return [(results[0][1][1], results[0][1][2])]

    def evaluate(self):
        #print ('######## Evaluating ########')
        probe_data = self.load_list_file(self.probe_list)
        correct_match = 0.0
        no_match = 0.0

        for i in tqdm(range(len(probe_data)),ncols=80):
            label = probe_data[i][0]
            image_name = probe_data[i][1]
            #name = '%s-%s'%(label, image_name)
            image_path = os.path.join(self.data_root, label, image_name + IMAGE_TYPE) 
            #print('Evaluating probe image: %s ...' % image_path)

            #img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            names = self.recognition(img)

            if len(names) > 0:
                best_match = names[0][0]
                best_prob = names[0][1]
                #print ('Best Match is %s with the probability of %f ' % (best_match, best_prob))
                if best_match.split('+')[0] == label:
                    correct_match += 1
            else:
                no_match += 1
                best_match = 'NaN'
                #print ('Cannot find the similar image in the gallery!')
            

        acc = correct_match / (len(probe_data) - no_match)

        print ('########  Evaluation Summary ########')
        print ('* Evaluated %d probe images in total.' % len(probe_data))
        print ('* %d probe images found matches in the gallery.' % (len(probe_data) - no_match))
        print ('* Among all matched probe images, %d images got matched correctly. The accuracy is %f.' % (correct_match, acc))
        print ('* %d images found no match in the gallery.' % no_match)
    
    def evaluate_image(self, image_path):

        print('Evaluating probe image: %s ...' % image_path)
        p_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
        names = self.recognition(p_img)
        
        #cv2.imshow("probe image", img)
        #cv2.waitKey(1)
        #cv2.destroyAllWindows()
        #print (names)
        if len(names) > 0:
            best_match = names[0][0]
            best_prob = names[0][1]

            ax  = plt.subplot(121)
            p_img = p_img[:, :, ::-1]
            plt.imshow(p_img)
            plt.axis('off')
            ax.set_title('Probe Image')
            
            ax = plt.subplot(122)
            res_label = best_match.split('+')[0]
            res_image = best_match.split('+')[1] 
            res_path = os.path.join(self.data_root, res_label, res_image + IMAGE_TYPE)
            g_img = cv2.imdecode(np.fromfile(res_path, dtype=np.uint8), 1)

            g_img = g_img[:, :, ::-1]
            plt.imshow(g_img)
            plt.axis('off')
            ax.set_title('Best Match Image')
            plt.show()

            print ('Best Match is %s with the probability of %f ' % (best_match, best_prob))


        else:
            g_img = None
            print ('Cannot find the similar image in the gallery!')



if __name__ == '__main__':
    test = FaceEval()
    test.update_face_data()
    cap = cv2.VideoCapture('test.mp4')
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            boxes, names = test.recognition(img)
            print(names)
            img = test.draw_face(img, boxes, names)
