from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from pathlib import Path
from collections import Counter

def inspect_mcap_with_rosbags(bag_path):
    bag_path = Path(bag_path)
    print(f"🔍 rosbags로 MCAP 분석 시작: {bag_path}")
    print("=" * 90)

    with Reader(bag_path) as reader:
        # 토픽 및 연결 정보
        print("📋 발견된 토픽 목록:")
        topic_info = {}
        for conn in reader.connections:
            topic_info[conn.topic] = {
                'msgtype': conn.msgtype,
                'count': 0
            }
            print(f"   • {conn.topic:<45} | {conn.msgtype}")

        # 메시지 카운트 및 샘플링
        print("\n📊 메시지 처리 중...")
        topic_counter = Counter()
        total_msgs = 0
        sample_messages = []

        for connection, timestamp, rawdata in reader.messages():
            topic_counter[connection.topic] += 1
            total_msgs += 1

            if len(sample_messages) < 5 and connection.topic not in [s[0] for s in sample_messages]:
                try:
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    sample_messages.append((connection.topic, type(msg).__name__, timestamp))
                except:
                    pass

            if total_msgs > 500:   # 너무 많으면 제한
                break

        print(f"\n📈 총 메시지 수: {total_msgs}개 (처음 {min(500, total_msgs)}개 기준)")
        for topic, count in topic_counter.most_common():
            print(f"   • {topic:<45} : {count:6d}개")

        print("\n🧪 샘플 메시지 (첫 5개 토픽):")
        for topic, msg_type, ts in sample_messages:
            print(f"   • {topic} → {msg_type} (timestamp: {ts})")

    print("=" * 90)
    print("검사 완료!")


# ====================== 사용 예시 ======================
if __name__ == "__main__":
    # 당신의 rosbag2 폴더 경로 또는 mcap 파일 경로를 넣으세요
    BAG_PATH = "/home/affctiv/openvla-oft/my_dataset/rosbag2_2026_05_08-15_11_07"   # ← 수정
    
    if Path(BAG_PATH).exists():
        inspect_mcap_with_rosbags(BAG_PATH)
    else:
        print("❌ 경로를 찾을 수 없습니다.")