

class PATHS:
    MODEL_CHECKPOINT = 'dataset/models/checkpoints'
    DATA = 'dataset/IMU_Mocap_Output_Data'
    TRAIN_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/train_data_label.pt'
    TEST_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/test_data_label.pt'
    ACL_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/acl_data_label.pt'
    HEALTHY_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/healthy_data_label.pt'
    ALL_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/all_data_label.pt'
    EXTRA_SUBJECTS = 'dataset/IMU_Mocap_Output_Data/extra_subjects.pt'


class DATA:
    IMU_LIST = ['left lateral shank', # 'left posterior shank',
                'left lateral thigh', # 'left posterior thigh',
                'right lateral shank', # 'right posterior shank',
                'right lateral thigh',] # 'right posterior thigh',]

    #JOINT_LIST = ['lhip', 'lknee', 'rhip', 'rknee']
    JOINT_LIST = [ 'lknee',  'rknee']
    JOINT_IMU_MAPPER = {
     #   'lhip': ['sacrum', 'left lateral thigh'],
        'lknee': ['left lateral thigh', 'left lateral shank'],
     #   'rhip': ['sacrum', 'right lateral thigh'],
        'rknee': ['right lateral thigh', 'right lateral shank']}

