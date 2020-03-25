import os
import cv2
import json 
from random import randint
import pickle

root_dir = "data/dewarp_labeled_data"

prepared_annotations = []

for f in os.listdir(root_dir):
    if "Invoice" not in f:
        continue

    if "Testing" in f or "Scanned" in f:
        continue

    folder_set = root_dir + "/" + f
    visual_dir = folder_set + "/" + "visualize"
    if not os.path.exists(visual_dir):
        os.mkdir(os.path.join(folder_set, "visualize"))


    for img_name in os.listdir(folder_set + "/images"):
        if not (img_name.endswith("JPG") or img_name.endswith("jpg") or img_name.endswith("png")):
            continue
        print(folder_set, img_name)
        with open(folder_set + "/labels/" + img_name.replace("JPG", "json")
        .replace("png", "json").replace("jpg", "json"), 'r', encoding='utf-8') as jsonfile:
            # print("Full path", folder_set + "/labels/" + img_name.replace("JPG", "json"))
            data = json.load(jsonfile)
            locations = data["attributes"]["_via_img_metadata"]["regions"]
            img = cv2.imread(folder_set + "/images/" + img_name)
            drawed_img = None

            B = randint(0, 255)
            G = randint(0, 255)
            R = randint(0, 255)

            more_than_two = 0
            for loc in locations:
                if loc["shape_attributes"]["name"] == "rect":
                    continue
                all_points_x = loc["shape_attributes"]["all_points_x"]
                all_points_y = loc["shape_attributes"]["all_points_y"]

                minx = min(all_points_x)
                maxx = max(all_points_x)
                miny = min(all_points_y)
                maxy = max(all_points_y)

                if (maxx - minx) * (maxy - miny) < 800000:
                    continue
                
                h, w, _ = img.shape
                if maxy > h:
                    print("Out of bot height", maxy - h)
                if maxx > w:
                    print("Out of right width", maxx - w)
                if minx < 0:
                    print("Out of top height",minx)
                if miny < 0:
                    print("Out of left width", miny)

                for i in range(len(all_points_x)):
                    if i+1 == len(all_points_x):
                        cv2.line(drawed_img, (int(all_points_x[i]), int(all_points_y[i])), (int(all_points_x[0]), int(all_points_y[0])), (B, G, R), 10)               
                        continue
                    cv2.line(drawed_img, (int(all_points_x[i]), int(all_points_y[i])), (int(all_points_x[i + 1]), int(all_points_y[i + 1])), (B, G, R), 10)               
                    break

            if len(all_points_x) == 4:
                more_than_two += 1

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
                net_input_size = 256
                for i in range(len(all_corners)):
                    all_corners[i] = int( 1.0 * net_input_size / padding_warped_image.shape[0] * all_corners[i])
                print(">>>>", all_corners)
                padding_warped_image = cv2.resize(padding_warped_image, (net_input_size, net_input_size))

                drawed_img = padding_warped_image.copy()
                cv2.line(drawed_img, (int(all_corners[0]), int(all_corners[1])), (int(all_corners[2]), int(all_corners[3])), (B, G, R), 10)
                cv2.line(drawed_img, (int(all_corners[2]), int(all_corners[3])), (int(all_corners[4]), int(all_corners[5])), (B, G, R), 10)
                cv2.line(drawed_img, (int(all_corners[4]), int(all_corners[5])), (int(all_corners[6]), int(all_corners[7])), (B, G, R), 10)
                cv2.line(drawed_img, (int(all_corners[6]), int(all_corners[7])), (int(all_corners[0]), int(all_corners[1])), (B, G, R), 10)
                # [[all_corners[idx][0], all_corners[idx][1]], [all_corners[idx][2], all_corners[idx][3]],
                #        [all_corners[idx][4], all_corners[idx][5]], [all_corners[idx][6], all_corners[idx][7]]]
                print(all_corners)
                corners = [[all_corners[0], all_corners[1]], [all_corners[2], all_corners[3]],
                        [all_corners[4], all_corners[5]], [all_corners[6], all_corners[7]]]

                annotation = {"img_paths": "000000502336.jpg", "img_width": 640, "img_height": 431, "num_keypoints": 8,
                            "segmentations": [],
                            "keypoints": [], "processed_other_annotations": []}
                annotation["img_paths"] = img_name
                annotation["img_height"] = padding_warped_image.shape[0]
                annotation["img_width"] = padding_warped_image.shape[1]
                print("After resized shape = ", padding_warped_image.shape)
                
                minx = min([x for i, x in enumerate(all_corners) if i % 2 == 0])
                maxx = max([x for i, x in enumerate(all_corners) if i % 2 == 0])

                miny = min([y for i, y in enumerate(all_corners) if i % 2 == 1])
                maxy = max([y for i, y in enumerate(all_corners) if i % 2 == 1])

                # Trung diem
                # corners = [[(maxx + minx) // 2, miny]
                # , [maxx, (maxy + miny) // 2]
                # , [(maxx + minx) // 2, maxy]
                # , [minx, (maxy + miny) // 2]]
                
                # x, y, width, hight
                bbx = [minx, miny, maxx - minx , maxy - miny]
                annotation['bbox'] = bbx
                annotation['objpos'] = [annotation['bbox'][0] + annotation['bbox'][2] / 2,
                                annotation['bbox'][1] + annotation['bbox'][3] / 2]
                annotation['scale_provided'] = annotation['bbox'][3] / net_input_size
                annotation['segmentations'] = [cor for cor in all_corners]

                MAX_POINT = 4
                keypoints = []
                for corner in corners:
                    keypoints.append([corner[0], corner[1], 1])
                for i in range(MAX_POINT - len(keypoints)):
                    keypoints.append([0, 0, 0])
                annotation["num_keypoints"] = MAX_POINT
                annotation["keypoints"] = keypoints
                prepared_annotations.append(annotation)

                # if not os.path.exists(folder_set + "/bimages/"):
                #     os.mkdir(folder_set + "/bimages/")
                folder_out = "D:/Coding/keypoint-detection/data/train_256_4/"
                cv2.imwrite(folder_out + img_name, padding_warped_image)
                # cv2.imwrite(folder_set + "/bimages/" + img_name, padding_warped_image)
            try:
                if drawed_img.shape:
                    cv2.imwrite(visual_dir + "/" + img_name, drawed_img)
            except Exception as e:
                print("Ex")
with open(folder_out + "prepared_train_annotation.pkl", 'wb') as f:
    pickle.dump(prepared_annotations, f)
    

            
