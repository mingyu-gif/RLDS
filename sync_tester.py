import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# 1. 타입스토어 설정 (bag metadata 기준: ROS2_JAZZY, 없으면 FOXY로 fallback)
typestore = get_typestore(getattr(Stores, 'ROS2_JAZZY', Stores.ROS2_FOXY))

# =========================================================================
# ur_msgs 타입을 rosbags가 기대하는 .msg 정의 기반으로 등록합니다.
# 기존 FIELDDEFS 직접 주입 방식은 rosbags 버전에 따라 타입 노드 형식이 달라져
# "(1, ('ur_msgs/msg/Digital', 0))" 같은 역직렬화 에러가 날 수 있습니다.
# =========================================================================
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
# =========================================================================

def get_digital_out_state(io_msg, pin):
    for digital in getattr(io_msg, 'digital_out_states', []):
        if digital.pin == pin:
            return 1.0 if digital.state else 0.0
    return None

def run_viewer(bag_path):
    # 최신 조인트 상태 및 그리퍼 상태 저장용 변수 (6번 인덱스 대용 변수 통합)
    current_joints = [0.0] * 6
    current_gripper = None
    current_gripper_source = "not found"
    has_valid_joints = False
    gripper_count = 0
    gripper_error_count = 0
    skipped_warmup_frames = 0
    
    with Reader(bag_path) as reader:
        print(f"\n[*] '{bag_path}' 데이터 읽기 시작")
        print("--------------------------------------------------")
        print(" [Enter] : 다음 프레임 + 조인트 및 그리퍼 값 출력")
        print(" [q]     : 종료")
        print("--------------------------------------------------")

        for connection, timestamp, rawdata in reader.messages():
            # 1. 조인트 토픽인 경우 값 업데이트
            if connection.topic == '/joint_states':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                if hasattr(msg, 'position') and len(msg.position) >= 6:
                    current_joints = msg.position[:6]
                    has_valid_joints = not np.all(np.array(current_joints, dtype=np.float32) == 0.0)

            # 2. 보내주신 바탕 코드를 적용한 그리퍼 데이터 처리 구간
            if connection.topic == '/io_and_status_controller/io_states':
                gripper_count += 1
                try:
                    # 타입 스토어 등록 덕분에 에러 없이 역직렬화 성공
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                    gripper_state = get_digital_out_state(msg, pin=0)
                    if gripper_state is not None:
                        current_gripper = gripper_state
                        current_gripper_source = "digital_out_states[pin=0].state"
                except Exception as e:
                    gripper_error_count += 1

            # 3. 이미지 토픽인 경우 화면에 표시하고 대기
            if connection.topic == '/camera/camera/color/image_raw':
                if not has_valid_joints or current_gripper is None:
                    skipped_warmup_frames += 1
                    continue

                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # 이미지 복구 (원본 크기)
                img_raw = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)

                # 화면에 타임스탬프 표시
                cv2.putText(img_bgr, f"Time: {timestamp / 1e9:.2f}s", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Bag Viewer (Press Enter to Next)", img_bgr)
                
                # 사용자 입력 대기
                key = cv2.waitKey(0)
                
                if key == 13: # Enter 키
                    # 터미널에 조인트 정보와 추출한 그리퍼 데이터(0번 핀 상태)를 함께 출력
                    print(f"T: {timestamp / 1e9:.2f}s | "
                          f"Joints: {[round(float(x), 4) for x in current_joints]} | "
                          f"Gripper: {current_gripper} ({current_gripper_source}, Count: {gripper_count}, Errors: {gripper_error_count}, "
                          f"SkippedWarmup: {skipped_warmup_frames})")
                elif key == ord('q'):
                    print("\n[*] 시청을 중단합니다.")
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    BAG_PATH = '/home/affctiv/openvla-oft/my_dataset/rosbag2_2026_06_01-18_35_01'
    run_viewer(BAG_PATH)
