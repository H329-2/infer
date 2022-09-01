   # print('---------------------------------------')
    # print(get)

    # for i in a[:, 5]:
    #     if flag_iou[i] != -1:
    #         cnt_nms = cnt_nms+1
    #         if i < ibeobj.shape[0]:
    #             k = obj.objects[i]
    #             cor = []
    #             cor1 = []
    #             for j in range(4):
    #                 us_tmp = [k.corners[j].x, k.corners[j].y, 0]
    #                 cor.append(us_tmp)
    #                 us_tmp = [k.corners[j].x, k.corners[j].y, 0.5]
    #                 cor1.append(us_tmp) 
    #             cors.append(cor)
    #             cors.append(cor1)
    #             label = np.append([2], label)
    #             score = np.append([2], score)
    #         else:
    #             t = torch.cat((corners_1[i - ibeobj.shape[0]], t), dim=0)
    #             label = np.append(bbox_results['lables'][i-ibeobj.shape[0]], label)
    #             score = np.append(bbox_results['scores'][i-ibeobj.shape[0]], score)
    #         for j in range(git.shape[1]):
    #             if j!=i and flag_iou[j]!=-1 and git[i, j] > 0.01:
    #                 flag_iou[j] = -1

    # t = t.reshape(-1, 8, 3)
    # print(flag_iou)
    # print(cnt_nms)

    # cors = np.array(cors).astype(np.float32)
    # cors = torch.from_numpy(cors).cuda()
    # cors = cors.reshape(-1, 3)
    # cors = cors @ tf_array1 + xyz_array1
    # cors = cors.reshape(-1, 8, 3)
    # cors = torch.cat((cors, t), dim=0)
    # print(cors.shape[0])
    # print(label.shape)
    # print(score.shape)