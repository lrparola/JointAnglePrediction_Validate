

class PATHS:
    MODEL_CHECKPOINT = 'dataset/models/checkpoints'
    DATA = 'dataset/IMU_Mocap_Output_Data'
    TRAIN_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/train_data_label.pt'
    TEST_DATA_LABEL = 'dataset/IMU_Mocap_Output_Data/test_data_label.pt'


class DATA:
    IMU_LIST = ['left lateral shank', # 'left posterior shank',
                'left lateral thigh', # 'left posterior thigh',
                'right lateral shank', # 'right posterior shank',
                'right lateral thigh', # 'right posterior thigh',
                'sacrum']

    JOINT_LIST = ['lhip', 'lknee', 'rhip', 'rknee']

    JOINT_IMU_MAPPER = {
        'lhip': ['sacrum', 'left lateral thigh'],
        'lknee': ['left lateral thigh', 'left lateral shank'],
        'rhip': ['sacrum', 'right lateral thigh'],
        'rknee': ['right lateral thigh', 'right lateral shank']}

