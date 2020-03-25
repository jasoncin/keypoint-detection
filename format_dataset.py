import os
import cv2
import json
from random import randint
import pickle

folder_set = "data/sample"
visual_dir = "data/sample/visualize"
prepared_annotations = []

for img_name in os.listdir(folder_set + "/images"):
    if not (img_name.endswith("JPG") or img_name.endswith("jpg") or img_name.endswith("png")):
        continue
    print(folder_set, img_name)
    json_file_name = folder_set + "/labels/" + img_name.replace("JPG", "json").replace("png", "json").replace("jpg", "json")
    if not os.path.exists(json_file_name):
        continue
    with open(json_file_name, 'r', encoding='utf-8') as jsonfile:
        # print("Full path", folder_set + "/labels/" + img_name.replace("JPG", "json"))
        data = json.load(jsonfile)
        locations = data["attributes"]["_via_img_metadata"]["regions"]
        img = cv2.imread(folder_set + "/images/" + img_name)
        drawed_img = None

        B = randint(0, 255)
        G = randint(0, 255)
        R = randint(0, 255)

        all_points_y = all_points_x = []
        full_anotation_linecut = []
        for loc in locations:
            if loc["shape_attributes"]["name"] == "rect":
                x,y,w,h = loc["shape_attributes"]["x"], loc["shape_attributes"]["y"], loc["shape_attributes"]["width"], loc["shape_attributes"]["height"]
                all_points_y = [y, y, y + h, y + h]
                all_points_x = [x, x + w, x + w, x]
            else:
                print(loc["shape_attributes"]["name"])
                all_points_x = loc["shape_attributes"]["all_points_x"]
                all_points_y = loc["shape_attributes"]["all_points_y"]

            minx = min(all_points_x)
            maxx = max(all_points_x)
            miny = min(all_points_y)
            maxy = max(all_points_y)

            h, w, _ = img.shape

            bordersize = 111
            if img.shape[0] > img.shape[1]:
                top_pad = bordersize
                lef_pad = (bordersize * 2 + img.shape[0] - img.shape[1]) // 2
            else:
                top_pad = (bordersize * 2 + img.shape[1] - img.shape[0]) // 2
                lef_pad = bordersize

            padding_warped_image = cv2.copyMakeBorder(
                img,
                top=top_pad,
                bottom=top_pad,
                left=lef_pad,
                right=lef_pad,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            all_corners = []
            for i in range(len(all_points_x)):
                all_corners.append(all_points_x[i] + lef_pad)
                all_corners.append(all_points_y[i] + top_pad)

            print(all_corners)
            net_input_size = 640
            for i in range(len(all_corners)):
                all_corners[i] = int(1.0 * net_input_size / padding_warped_image.shape[0] * all_corners[i])
            print(">>>>", all_corners)
            padding_warped_image = cv2.resize(padding_warped_image, (net_input_size, net_input_size))

            drawed_img = padding_warped_image.copy()
            cv2.line(drawed_img, (int(all_corners[0]), int(all_corners[1])), (int(all_corners[2]), int(all_corners[3])),
                     (B, G, R), 10)
            cv2.line(drawed_img, (int(all_corners[2]), int(all_corners[3])), (int(all_corners[4]), int(all_corners[5])),
                     (B, G, R), 10)
            cv2.line(drawed_img, (int(all_corners[4]), int(all_corners[5])), (int(all_corners[6]), int(all_corners[7])),
                     (B, G, R), 10)
            cv2.line(drawed_img, (int(all_corners[6]), int(all_corners[7])), (int(all_corners[0]), int(all_corners[1])),
                     (B, G, R), 10)
            # [[all_corners[idx][0], all_corners[idx][1]], [all_corners[idx][2], all_corners[idx][3]],
            #        [all_corners[idx][4], all_corners[idx][5]], [all_corners[idx][6], all_corners[idx][7]]]

            # cv2.imshow("Debug", drawed_img)
            # cv2.waitKey(0)

            print(all_corners)
            corners = [[all_corners[0], all_corners[1]], [all_corners[2], all_corners[3]],
                       [all_corners[4], all_corners[5]], [all_corners[6], all_corners[7]]]

            annotation = {"img_paths": "000000502336.jpg", "img_width": 640, "img_height": 431, "num_keypoints": 8, "segmentations": [], "keypoints": [], "processed_other_annotations": []}
            annotation["img_paths"] = img_name
            annotation["img_height"] = padding_warped_image.shape[0]
            annotation["img_width"] = padding_warped_image.shape[1]
            print("After resized shape = ", padding_warped_image.shape)

            minx = min([x for i, x in enumerate(all_corners) if i % 2 == 0])
            maxx = max([x for i, x in enumerate(all_corners) if i % 2 == 0])
            miny = min([y for i, y in enumerate(all_corners) if i % 2 == 1])
            maxy = max([y for i, y in enumerate(all_corners) if i % 2 == 1])

            # Trung diem
            corners = [[(maxx + minx) // 2, (maxy + miny) // 2]]

            # x, y, width, hight
            bbx = [minx, miny, maxx - minx, maxy - miny]
            annotation['bbox'] = bbx
            annotation['objpos'] = [annotation['bbox'][0] + annotation['bbox'][2] / 2,
                                    annotation['bbox'][1] + annotation['bbox'][3] / 2]
            annotation['scale_provided'] = annotation['bbox'][3] / net_input_size
            annotation['segmentations'] = [cor for cor in all_corners]

            MAX_POINT = 1
            keypoints = []
            for corner in corners:
                keypoints.append([corner[0], corner[1], 1])
            for i in range(MAX_POINT - len(keypoints)):
                keypoints.append([0, 0, 0])
            annotation["num_keypoints"] = MAX_POINT
            annotation["keypoints"] = keypoints

            # if not os.path.exists(folder_set + "/bimages/"):
            #     os.mkdir(folder_set + "/bimages/")
            folder_out = "data/sample/format/"
            if not os.path.exists(folder_out + img_name):
                cv2.imwrite(folder_out + img_name, padding_warped_image)
            # cv2.imwrite(folder_set + "/bimages/" + img_name, padding_warped_image)

            full_anotation_linecut.append(annotation)

        main_anotation = full_anotation_linecut[0]
        main_anotation["processed_other_annotations"] = full_anotation_linecut[1:]

        prepared_annotations.append(main_anotation)
        try:
            if drawed_img.shape:
                cv2.imwrite(visual_dir + "/" + img_name, drawed_img)
        except Exception as e:
            print("Ex")

with open(folder_set + "/format/" + "prepared_train_annotation.pkl", 'wb') as f:
    print(prepared_annotations)
    pickle.dump(prepared_annotations, f)



