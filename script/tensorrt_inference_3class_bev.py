#!/home/lumos/anaconda3/envs/mmdet3d/bin/python
import sys
# sys.path.append(r'/home/iair/Documents/Pioneer_ws/src/detection3d/script/model')
from cProfile import label
from operator import index
from mmcv import Config
import torch
from mmdeploy.utils import Backend
from mmcv import ops
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmdet3d.core.bbox.box_np_ops import boxes3d_to_corners3d_lidar
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
import rospy
from sensor_msgs.msg import PointCloud2
import message_filters
from ibeo_lidar_msgs.msg import object_filter_data 
import time
import tf
import ros_numpy
from cProfile import label
from operator import index
from turtle import pd
import torch
from spconv.pytorch.utils import PointToVoxel
from mmdeploy.utils import Backend
from typing import  Sequence, Union
import mmcv
from mmdet3d.models.builder import build_head
import torch.nn.functional as F
import time
from mmdeploy.codebase.base import BaseBackendModel

quesize = 10
rear = 1
front = 0

tf_queue = torch.from_numpy(np.zeros((quesize,3,3)).astype(np.float32)).cuda()
pc_queue = {}
for i in range(quesize):
    pc_queue[i]=''
xyz_queue = torch.from_numpy(np.zeros((quesize,3)).astype(np.float32)).cuda()
tmp_tf = torch.from_numpy(np.zeros((3,3)).astype(np.float32)).cuda()

tmp_xyz = torch.from_numpy(np.zeros(3).astype(np.float32)).cuda()
time_cnter = 0
num_cnter = 0



class FastVoxelDetectionModel:
    def __init__(self,
                 cfg_dict,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmcv.Config,
                 deploy_cfg: Union[str, mmcv.Config] = None):
        # 体素化，spconv实现
        self.cfg = cfg_dict
        self.voxel_generator = PointToVoxel(**cfg_dict.voxelization)
        # getbbox需要
        self.deploy_cfg = deploy_cfg
        head_cfg = dict(**model_cfg.model['pts_bbox_head'])
        head_cfg['train_cfg'] = None
        head_cfg['test_cfg'] = model_cfg.model['test_cfg']
        # print(head_cfg['test_cfg'])
        self.head = build_head(head_cfg)
        self.head.test_cfg = self.head.test_cfg['pts']

        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper (self, backend: Backend, backend_files: Sequence[str], device: str):
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=['/home/iair/Documents/mmdeploy/work_dirs/centerpoint_custom/part1.trt'],
            device=device,
            output_names=["onnx::Reshape_261"], 
            deploy_cfg=self.deploy_cfg)

        self.wrapper2 = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=['/home/iair/Documents/mmdeploy/work_dirs/centerpoint_custom/part2.trt'],
            device=device,
            output_names=["dir_scores", "bbox_preds", "scores"],
            deploy_cfg=self.deploy_cfg)

    def forward(self, points: Sequence[torch.Tensor], img_metas: Sequence[dict]):
        # 体素化 1ms左右
        voxels, coors, num_points = self.voxel_generator(points)
        input_dict = {
            'voxels': voxels,
            'num_points': num_points,
            'coors': F.pad(coors, pad=(1,0),value=0)
        }
        # 模型推理 8ms左右
        t1 =time.time()
        mid_outs = self.wrapper(input_dict)
        t2 = time.time()
        mid_outs['onnx::Reshape_261'][:, 0] = 0
        outs = self.wrapper2(mid_outs)
     
        # 获得bbox后处理 15->20ms
        bbox_results = self.get_bboxes(
            outs['scores'], outs['bbox_preds'], outs['dir_scores'], img_metas) 
        
        return bbox_results, t2-t1  

    def get_bboxes(self, cls_scores, bbox_preds, dir_scores, img_metas):
        rets = []
        scores_range = [0]
        bbox_range = [0]
        dir_range = [0]
        for i in range(len(self.cfg.head.tasks)):
            scores_range.append(scores_range[i] + self.head.num_classes[i])
            bbox_range.append(bbox_range[i] + 4)
            dir_range.append(dir_range[i] + 2)
        for task_id in range(len(self.head.num_classes)):
            num_class_with_bg = self.head.num_classes[task_id]
            batch_heatmap = cls_scores[:, scores_range[task_id]:scores_range[task_id + 1],
                            ...].sigmoid()
            batch_reg = bbox_preds[:,
                        bbox_range[task_id]:bbox_range[task_id] + 2,
                        ...]
            shape = batch_reg.shape
            batch_hei = 2 * batch_reg.new_ones(shape[0], 1, *shape[2:])
            batch_dim = torch.exp(bbox_preds[:, bbox_range[task_id] +
                                                   2:bbox_range[task_id] + 4,
                                      ...])
            hei_dim = 0.5 * batch_reg.new_ones(shape[0], 1, *shape[2:])
            batch_dim = torch.cat((batch_dim, hei_dim), dim=1)

            batch_vel = batch_reg.new_zeros(shape[0], 2, *shape[2:])

            batch_rots = dir_scores[:,
                         dir_range[task_id]:dir_range[task_id + 1],
                         ...][:, 0].unsqueeze(1)
            batch_rotc = dir_scores[:,
                         dir_range[task_id]:dir_range[task_id + 1],
                         ...][:, 1].unsqueeze(1)
            temp = self.box_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]

            rets.append(
                self.head.get_task_detections(num_class_with_bg, batch_cls_preds,
                                              batch_reg_preds, batch_cls_labels,
                                              img_metas[0]))
        for k in rets[0][0].keys():
            if k == 'bboxes':
                bboxes = torch.cat([ret[0][k] for ret in rets])
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            elif k == 'scores':
                scores = torch.cat([ret[0][k] for ret in rets])
            elif k == 'labels':
                flag = 0
                for j, num_class in enumerate(self.head.num_classes):
                    rets[j][0][k] += flag
                    flag += num_class
                labels = torch.cat([ret[0][k].int() for ret in rets])
        valid = bboxes[:, 0:2].norm(p=2, dim=1) >= 1.5  # remove bboxes in bottom center
        return {'bboxes': bboxes[valid].cpu().numpy(),
                'scores': scores[valid].cpu().numpy(),
                'lables': labels[valid].cpu().numpy()}    

def getTransArray(x,y,z,w,x0,y0,z0,temp,xyz):
    temp[0][0]= 1-2*(y*y+z*z)
    temp[0][1]= 2*(x*y+z*w)
    temp[0][2]= 2*(x*z-y*w)
    temp[1][0]= 2*(x*y-z*w)
    temp[1][1]= 1-2*(x*x+z*z)
    temp[1][2]= 2*(y*z+x*w)
    temp[2][0]= 2*(x*z+y*w)
    temp[2][1]= 2*(y*z-x*w)
    temp[2][2]= 1-2*(x*x+y*y)
    xyz[0] = x0
    xyz[1] = y0
    xyz[2] = z0

def reprocess(bbox_results, obj, corners):
    global tmp_tf, tmp_xyz
    # 设定ibeo的分数
    ibeoscore = 0.5
    '''
       3----0
      /    /|
     /    / | y
    2----1---
           x
    '''
    corners_1 = torch.from_numpy(corners).cuda()
    corners = corners_1.reshape(-1,3) @ tmp_tf.T + tmp_xyz
    corners = corners.reshape(-1, 8, 3)
    src1 = torch.from_numpy(bbox_results['bboxes'][:, 3]).reshape([-1, 1]).cuda()
    src2 = torch.from_numpy(bbox_results['bboxes'][:, 4]).reshape([-1, 1]).cuda()
    src3 = torch.from_numpy(bbox_results['scores']).reshape([-1, 1]).cuda()
    src = torch.cat((src1, src2), dim = 1)

    # 通过角点来得到朝向和中心点， 通过bbox直接获取长宽
    tmp = torch.cat((1/2*(corners[:, 0, :2]+corners[:, 2, :2]), src), dim = 1)
    src = corners[:, 0, :2]- corners[:, 1, :2]
    tmp = torch.cat((tmp, torch.atan(src[:, 1]/src[:, 0]).reshape(-1,1)), dim=1)
    tmp = torch.cat((tmp, src3), dim=1)
    # 获取ibeo bboxes
    ibeobj = []
    ibeocor = []
    for i in obj.objects:
        us_tmp = [i.center.x, i.center.y, i.length, i.width, i.orientation, ibeoscore]
        us1_tmp = [i.center.x, i.center.y, 0, i.length, i.width, 0.5, i.orientation]
        ibeobj.append(us_tmp)
        ibeocor.append(us1_tmp)

    ibeobj = np.array(ibeobj).astype(np.float32)
    ibeobj = torch.from_numpy(ibeobj).cuda()
    
    ibeocor = np.array(ibeocor).astype(np.float32)
    ibeocor = boxes3d_to_corners3d_lidar(ibeocor)
    ibeocor = torch.from_numpy(ibeocor).cuda() 
    ibeocor = ibeocor@tf_queue[rear-1]+xyz_queue[rear-1]

    # git = ops.box_iou_rotated(ibeobj, tmp, aligned=True)
    lable = np.zeros(ibeobj.shape[0]).astype(np.int32)
    lable = np.append(lable, bbox_results['lables'], axis=0)
    obj_out = torch.cat((ibeobj, tmp), dim=0)
    obj_out = torch.reshape(obj_out, [-1, 6])

    cors = torch.cat((ibeocor, corners_1), dim = 0)
    tensor, index = ops.nms_rotated(obj_out[:, :5], obj_out[:, 5], 0.3)
 
    return index, cors, lable, obj_out[:, 5], ibeobj.shape[0]

def callback_point(msg, ibeobj):
    global rear, front, tmp_tf, tmp_xyz, tf_queue, xyz_queue, quesize
    '''
    实现模型推理， 总时长在40ms左右 发布bbox耗时会更长
    '''
    t3 = time.time() 
    time_fr = msg.header.stamp
    # 得到当前帧对应的tf 2ms
    (transcar,rotcar) = tf_listener.lookupTransform('world', 'velo_middle', rospy.Time(0))
    getTransArray(rotcar[0], rotcar[1], rotcar[2], rotcar[3] ,transcar[0], transcar[1], transcar[2], tf_queue[rear], tf_queue[rear])
    (transw, rotw) = tf_listener.lookupTransform('velo_middle', 'world', rospy.Time(0))
    getTransArray(rotw[0], rotw[1], rotw[2], rotw[3] ,transw[0], transw[1], transw[2], tmp_tf, tmp_xyz)
    
    # 读取点云转为tensor < 1ms
    pc = ros_numpy.numpify(msg)
    points = np.zeros((pc.shape[0], 3))
    points[:, 0] = pc['x']
    points[:, 1] = pc['y']
    points[:, 2] = pc['z']
    points = points.astype(np.float32)
    points = torch.from_numpy(points[:, :3]).contiguous().cuda()
    t_1 = time.time()
    pc_queue[rear] = points.clone()
    
    # 多帧点云合并
    if (rear == front):
        # 对于存储的点云，只要不是当前点云，就将它进行转换并且输出到final中
        for cnt in range(quesize):
            if cnt != rear:
                # 计算坐标变换之后的矩阵
                pc_qt = pc_queue[cnt] @ tf_queue[cnt].T + xyz_queue[cnt]
                pc_qt = pc_qt @ tmp_tf.T + tmp_xyz
                points = torch.cat([points,pc_qt],dim=0)
                # print(points.shape[0])
    if rear == front:
        front = (front +1)%quesize 
    rear=(rear+1) % quesize
    t_2 = time.time()

    # 得到bbox  25 -> 30 ms
    bbox_results, time_s = fast_deploy_model.forward(points, img_metas)
    
    # 得到角点 < 1ms
    corners = boxes3d_to_corners3d_lidar(bbox_results['bboxes'][:, :7])
    
    # 后融合 5ms
    index, cors, lable, score, ibeo_len = reprocess(bbox_results, ibeobj, corners)
    print('*'*50)
    print(t_2 - t_1)
    
    # 发布bbox 耗时很高需要70ms
    # fast_deploy_model.publish_bboxes(corners, labels = bbox_results['lables'], scores = bbox_results['scores'])
    t2 = time.time()
    print(time_s)
    
def main():
    global fast_deploy_model, tf_listener, img_metas, point_publisher, colors, lines, topic_pub_box, topic_sub_image, ibeo_publisher
    rospy.init_node('object_detector', anonymous=True)

    deploy_path = '/home/lumos/mmdeploy/work_dirs/work_823/end2end.engine'
    det3d_cfg_path = '/home/lumos/文档/mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus_custom.py'
    deploy_cfg_path = '/home/lumos/mmdeploy/configs/mmdet3d/voxel-detection/voxel-detection_tensorrt_dynamic-nus.py'
    set_device = 'cuda:0'
    topic_sub_point = '/lidar/vlp32_middle/PointCloud2_compensated'
    topic_sub_image = '/camera/image_color/Image'
    topic_sub_ibeo = '/perception/objects'
    topic_pub_box = '/detected_boxes3d'
    topic_pub_point = '/point_cloud_pub'
    topic_pub_iobj = '/iobj_box3d'

    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
             [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
              [1, 1, 1], [0.25, 0.25, 0.25], [65/255, 105/255, 225/255], [160/255, 32/255, 240/255]]

    point_publisher = rospy.Publisher(topic_pub_point, PointCloud2, queue_size=50)
    ibeo_publisher = rospy.Publisher(topic_pub_iobj, MarkerArray, queue_size=10)
    img_metas = [[dict(box_type_3d=LiDARInstance3DBoxes)]]
    det3d_cfg = Config.fromfile(det3d_cfg_path)
    deploy_cfg = Config.fromfile(deploy_cfg_path)
    fast_deploy_model = FastVoxelDetectionModel(backend=Backend.TENSORRT, backend_files=[deploy_path],
                                                device=set_device,
                                                model_cfg=det3d_cfg, deploy_cfg=deploy_cfg)

    tf_listener = tf.TransformListener()

    sub = message_filters.Subscriber(topic_sub_point, PointCloud2)
    sub_img = message_filters.Subscriber(topic_sub_ibeo, object_filter_data)
    ts = message_filters.ApproximateTimeSynchronizer([sub, sub_img], 10, 0.1, allow_headerless=True)

    ts.registerCallback(callback_point)

    rospy.spin()

if __name__ == '__main__':
    main()
