import cv2
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from typing import Iterator, Tuple, Any

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

typestore = get_typestore(getattr(Stores, 'ROS2_JAZZY', Stores.ROS2_FOXY))

UR_MSG_DEFS = {
    'ur_msgs/msg/Digital': """
uint8 pin
bool state
""",
    'ur_msgs/msg/Analog': """
uint8 CURRENT=0
uint8 VOLTAGE=1

uint8 pin
uint8 domain
float32 state
""",
    'ur_msgs/msg/IOStates': """
Digital[] digital_in_states
Digital[] digital_out_states
Digital[] flag_states
Analog[] analog_in_states
Analog[] analog_out_states
""",
}


for msgtype, msgdef in UR_MSG_DEFS.items():
    typestore.register(get_types_from_msg(msgdef, msgtype))


def get_digital_out_state(io_msg, pin):
    for digital in getattr(io_msg, 'digital_out_states', []):
        if digital.pin == pin:
            return 1.0 if digital.state else 0.0
    return None

class Ur5ePickAndPlaceDataset(tfds.core.GeneratorBasedBuilder):
    """UR5e Pick and Place RLDS Dataset Builder."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release with UR5e and RealSense(848x480).',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(shape=(224, 224, 3), dtype=np.uint8),
                        'state': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    }),
                    'action': tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                    'language_instruction': tfds.features.Text(),
                    'language_embedding': tfds.features.Tensor(shape=(512,), dtype=np.float32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(
                path='/home/affctiv/openvla-oft/my_dataset/rosbag2_*'
            ),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:

        def _parse_example(episode_path):
            episode = []
            current_state = np.zeros(7, dtype=np.float32)
            current_gripper = None
            has_valid_joints = False
            
            # 🔹 스테이지 추적 및 빌더 내부 검증용 변수
            stage = 0 
            last_printed_stage = -1
            stage_step_counts = {0: 0, 1: 0, 2: 0}  # 각 스테이지별 저장된 실제 Step 수 카운트

            print(f"\n[DEBUG] 파일 처리 시작: {episode_path}")
            joint_count = joint_zero_count = joint_error_count = 0
            camera_count = camera_joint_zero_count = 0
            gripper_count = gripper_error_count = 0
            gripper_not_found_count = camera_gripper_none_count = 0
            gripper_zero_count = 0
            gripper_one_count = 0

            with Reader(episode_path) as reader:
                for connection, timestamp, rawdata in reader.messages():
                    # 1. Joint States 처리
                    if connection.topic == '/joint_states':
                        joint_count += 1
                        try:
                            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                            if hasattr(msg, 'position') and len(msg.position) >= 6:
                                current_state[:6] = np.array(msg.position[:6], dtype=np.float32)
                                if np.all(current_state[:6] == 0.0):
                                    has_valid_joints = False
                                    joint_zero_count += 1
                                    continue
                                has_valid_joints = True
                        except Exception:
                            joint_error_count += 1
                            continue
                        continue

                    # 2. Gripper (IOStates) 처리
                    if connection.topic == '/io_and_status_controller/io_states':
                        gripper_count += 1
                        try:
                            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                            gripper_state = get_digital_out_state(msg, pin=0)
                            if gripper_state is not None:
                                current_gripper = gripper_state
                                if gripper_state == 1.0:
                                    gripper_one_count += 1
                                else:
                                    gripper_zero_count += 1
                            else:
                                gripper_not_found_count += 1

                        except Exception as e:
                            gripper_error_count += 1
                        continue

                    # 3. Camera Image 처리
                    if connection.topic == '/camera/camera/color/image_raw':
                        camera_count += 1
                        try:
                            if not has_valid_joints:
                                camera_joint_zero_count += 1
                                continue
                            if current_gripper is None:
                                camera_gripper_none_count += 1
                                continue

                            # 1) 그리퍼 상태에 따라 스테이지(숫자) 상태 머신을 먼저 전환
                            actual_gripper_proprio = round(float(current_gripper))

                            if stage == 0 and actual_gripper_proprio == 0.0:
                                stage = 1
                            elif stage == 1 and actual_gripper_proprio == 1.0:
                                stage = 2

                            # 2) 현재 최종 확정된 stage 상태를 기준으로 텍스트 매핑 (밀림 방지)
                            if stage == 0:
                                task = "pick object"
                            elif stage == 1:
                                task = "place object"
                            elif stage == 2:
                                task = "move to initial position"

                            # 인스트럭션 텍스트 생성
                            instruction = f"In: What action should the robot take to {task}?\nOut:"

                            # [빌더 내부 스테이지 전환 감지 로그] -> 이제 정확히 매칭되어 찍힙니다.
                            if stage != last_printed_stage:
                                print(f"    ↳ 🔔 [빌더 내부 스테이지 전환 감지] Stage: {stage} | Gripper 원본값: {current_gripper} -> round: {actual_gripper_proprio}")
                                print(f"    ↳ 📝 적용 문장: '{task}'")
                                last_printed_stage = stage

                            # 현재 스테이지에 해당하는 Step 카운트 1 증가
                            stage_step_counts[stage] += 1

                            # 언어 모델 임베딩 생성
                            language_embedding = self._embed([instruction])[0].numpy()

                            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                            height = msg.height
                            width = msg.width
                            
                            image_array = np.frombuffer(msg.data, dtype=np.uint8)
                            image_array = image_array.reshape((height, width, 3))
                            
                            min_side = min(height, width)
                            start_x = (width - min_side) // 2
                            start_y = (height - min_side) // 2
                            image_array = image_array[start_y:start_y+min_side, start_x:start_x+min_side]
                            image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_CUBIC)

                            current_state[6] = current_gripper

                            episode.append({
                                'observation': {
                                    'image': image_array,
                                    'state': current_state.copy(),
                                },
                                'action': current_state.copy(),
                                'discount': 1.0,
                                'reward': 0.0,
                                'is_first': len(episode) == 0,
                                'is_last': False,
                                'is_terminal': False,
                                'language_instruction': instruction,
                                'language_embedding': language_embedding,
                            })
                        except Exception as e:
                            print(f"[IMAGE ERROR] 이미지 파싱 실패 원인: {e}")
                            continue
                        continue

            step_gripper_values = [float(step['observation']['state'][6]) for step in episode]
            step_gripper_zero_count = step_gripper_values.count(0.0)
            step_gripper_one_count = step_gripper_values.count(1.0)

            print(
                f"[DEBUG] 완료 → Joint: {joint_count}개, "
                f"JointZero: {joint_zero_count}개, JointError: {joint_error_count}개, "
                f"Gripper: {gripper_count}개, GripperError: {gripper_error_count}개, "
                f"Camera: {camera_count}개, Step: {len(episode)}개"
            )
            # 💡 [체킹 코드 추가]: 최종 요약 출력 시 각 스테이지별로 실제 담긴 Step(프레임) 수를 리포트팅합니다.
            print(
                f"[DEBUG] 📊 스테이지별 생성된 Step 수 → [Stage 0(Pick)]: {stage_step_counts[0]}개 | "
                f"[Stage 1(Place)]: {stage_step_counts[1]}개 | [Stage 2(Initial)]: {stage_step_counts[2]}개"
            )
            print(
                f"[DEBUG] Camera skip → Joint state 아직 0인 프레임: "
                f"{camera_joint_zero_count}개"
            )
            print(
                f"[DEBUG] Gripper IO 값 → 0: {gripper_zero_count}개, "
                f"1: {gripper_one_count}개, None: {gripper_not_found_count}개"
            )
            print(
                f"[DEBUG] Gripper Step 값(state[6]) → 0: {step_gripper_zero_count}개, "
                f"1: {step_gripper_one_count}개"
            )
            print(
                f"[DEBUG] Camera skip → Gripper state 아직 None인 프레임: "
                f"{camera_gripper_none_count}개"
            )

            # Action shift
            if len(episode) > 1:
                for i in range(len(episode) - 1):
                    episode[i]['action'] = episode[i+1]['observation']['state'].copy()

            if len(episode) > 0:
                episode[-1]['action'] = episode[-1]['observation']['state'].copy()
                episode[-1]['is_last'] = True
                episode[-1]['is_terminal'] = True
                episode[-1]['reward'] = 1.0

            return episode_path, {
                'steps': episode,
                'episode_metadata': {'file_path': episode_path}
            }

        episode_paths = glob.glob(path)
        for sample in episode_paths:
            yield _parse_example(sample)