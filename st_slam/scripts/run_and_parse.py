#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import argparse

def parse_log_output(log_content):
    metrics = {
        'ate_rmse': None,
        'ate_mean': None,
        'rpe_trans_rmse': None,
        'rpe_rot_rmse': None,
        'loops_detected': None,
        'tracking_success': None,
        'num_keyframes': None,
        'total_frames': None,
        'wall_time': None,
        'edges_built': None,
        'loop_edges': None,
        'pgo_correction': None,
        'final_error': None,
        'iterations': None,
        'drift_correction': None,
        'ate_quality': None,
        'mean_time_ms': None
    }
    
    # 从实际日志中提取信息
    loop_closed_matches = re.findall(r'\[LOOP_CLOSED\]', log_content)
    metrics['loops_detected'] = len(loop_closed_matches)
    
    # 提取ATE RMSE
    ate_section = re.search(r'--- ATE \(Absolute Trajectory Error\) ---(.*?)---', log_content, re.DOTALL)
    if ate_section:
        ate_rmse_match = re.search(r'RMSE:\s+([\d.]+)\s+m', ate_section.group(1))
        if ate_rmse_match:
            metrics['ate_rmse'] = float(ate_rmse_match.group(1))
        ate_mean_match = re.search(r'Mean:\s+([\d.]+)\s+m', ate_section.group(1))
        if ate_mean_match:
            metrics['ate_mean'] = float(ate_mean_match.group(1))
    
    # 提取RPE信息
    rpe_section = re.search(r'--- RPE \(Relative Pose Error\) ---(.*?)---', log_content, re.DOTALL)
    if rpe_section:
        rpe_t_match = re.search(r'Translation RMSE:\s+([\d.]+)\s+m', rpe_section.group(1))
        if rpe_t_match:
            metrics['rpe_trans_rmse'] = float(rpe_t_match.group(1))
        rpe_r_match = re.search(r'Rotation RMSE:\s+([\d.]+)\s+deg', rpe_section.group(1))
        if rpe_r_match:
            metrics['rpe_rot_rmse'] = float(rpe_r_match.group(1))
    
    # 提取ATE质量
    quality_match = re.search(r'Quality Assessment:\s+\[([^\]]+)\]', log_content)
    if quality_match:
        metrics['ate_quality'] = quality_match.group(1)
    
    # 提取Loops Detected
    loops_detected_match = re.search(r'Loops Detected:\s+(\d+)', log_content)
    if loops_detected_match:
        metrics['loops_detected'] = int(loops_detected_match.group(1))
    
    # 提取Tracking Success Rate
    success_match = re.search(r'Tracking Success Rate:\s+([\d.]+)%', log_content)
    if success_match:
        metrics['tracking_success'] = float(success_match.group(1))
    
    # 提取Total Frames
    frames_match = re.search(r'Total Frames:\s+(\d+)', log_content)
    if frames_match:
        metrics['total_frames'] = int(frames_match.group(1))
    
    # 提取最后一次PGO的修正值
    correction_matches = re.findall(r'Applied correction: trans=([\d.]+)m rot=([\d.]+)deg', log_content)
    if correction_matches:
        metrics['pgo_correction'] = float(correction_matches[-1][0])
    
    # 提取漂移信息（如果有）
    drift_match = re.search(r'drift=([\d.]+)cm ([\d.]+)deg', log_content)
    if drift_match:
        metrics['drift_correction'] = (float(drift_match.group(1)), float(drift_match.group(2)))
    
    # 提取关键帧数量
    kf_matches = re.findall(r'on (\d+) keyframes', log_content)
    if kf_matches:
        metrics['num_keyframes'] = int(kf_matches[-1])
    
    # 提取运行时间
    wall_time_match = re.search(r'Wall time:\s*([\d.]+)\s*s', log_content)
    if wall_time_match:
        metrics['wall_time'] = float(wall_time_match.group(1))
    
    # 提取平均处理时间
    mean_time_match = re.search(r'Mean:\s+([\d.]+)\s+ms', log_content)
    if mean_time_match:
        metrics['mean_time_ms'] = float(mean_time_match.group(1))
    
    # 提取边数量（最后一次PGO）
    edges_matches = re.findall(r'Built (\d+) edges \((\d+) odometry, (\d+) loop\)', log_content)
    if edges_matches:
        metrics['edges_built'] = int(edges_matches[-1][0])
        metrics['loop_edges'] = int(edges_matches[-1][2])
    
    # 提取BuildFromKeyframes中保留的回环边
    kept_edges_match = re.findall(r'BuildFromKeyframes: kept (\d+) loop edges', log_content)
    if kept_edges_match:
        metrics['loop_edges'] = int(kept_edges_match[-1])
    
    return metrics

def print_metrics(metrics):
    print("="*70)
    print("          ST-SLAM 4.0 - 关键指标解析结果")
    print("="*70)
    
    print("\n📊 精度指标 (ATE - Absolute Trajectory Error)")
    print("-" * 45)
    if metrics['ate_rmse'] is not None:
        quality = "🏆 SOTA" if metrics['ate_rmse'] < 0.01 else \
                  "✨ Excellent" if metrics['ate_rmse'] < 0.015 else \
                  "👍 Good" if metrics['ate_rmse'] < 0.03 else \
                  "⚠️ Acceptable" if metrics['ate_rmse'] < 0.06 else \
                  "❌ Needs Work"
        print(f"  ATE RMSE:    {metrics['ate_rmse']*100:.2f} cm  [{quality}]")
        if metrics['ate_mean'] is not None:
            print(f"  ATE Mean:    {metrics['ate_mean']*100:.2f} cm")
        if metrics['ate_quality']:
            print(f"  Quality Tag: [{metrics['ate_quality']}]")
    
    print("\n📈 局部精度指标 (RPE - Relative Pose Error)")
    print("-" * 45)
    if metrics['rpe_trans_rmse'] is not None:
        print(f"  Translation RMSE: {metrics['rpe_trans_rmse']*100:.2f} cm")
    if metrics['rpe_rot_rmse'] is not None:
        print(f"  Rotation RMSE:    {metrics['rpe_rot_rmse']:.2f} deg")
    
    print("\n🔄 回环检测与位姿图优化 (Loop Closure & PGO)")
    print("-" * 45)
    if metrics['loops_detected'] is not None:
        status = "✅ 回环检测成功！" if metrics['loops_detected'] > 0 else "❌ 未检测到回环"
        print(f"  Loops Detected:   {metrics['loops_detected']}  {status}")
    if metrics['loop_edges'] is not None:
        print(f"  Loop Edges in PGO: {metrics['loop_edges']}")
    if metrics['pgo_correction'] is not None:
        print(f"  PGO Translation Correction: {metrics['pgo_correction']*100:.2f} cm")
    if metrics['edges_built'] is not None:
        print(f"  Total PoseGraph Edges: {metrics['edges_built']}")
    if metrics['drift_correction'] is not None:
        print(f"  Drift at Loop Close: {metrics['drift_correction'][0]:.1f} cm, {metrics['drift_correction'][1]:.1f} deg")
    
    print("\n📁 数据统计")
    print("-" * 45)
    if metrics['total_frames'] is not None:
        print(f"  Total Frames:     {metrics['total_frames']}")
    if metrics['num_keyframes'] is not None:
        print(f"  Keyframes:        {metrics['num_keyframes']}")
    if metrics['tracking_success'] is not None:
        print(f"  Tracking Success: {metrics['tracking_success']:.1f}%")
    
    print("\n⏱️ 性能指标")
    print("-" * 45)
    if metrics['wall_time'] is not None:
        fps = metrics['total_frames'] / metrics['wall_time'] if metrics['total_frames'] else 0
        print(f"  Wall Time:        {metrics['wall_time']:.2f} s")
        print(f"  FPS:              {fps:.1f}")
    if metrics['mean_time_ms'] is not None:
        print(f"  Avg Time/Frame:   {metrics['mean_time_ms']:.2f} ms")
    
    print("\n" + "="*70)
    
    print("\n📋 评估总结")
    print("-" * 45)
    if metrics['ate_rmse'] is not None and metrics['loops_detected'] is not None:
        if metrics['ate_rmse'] < 0.06 and metrics['loops_detected'] > 0:
            print("✅ 系统运行正常，回环检测生效！")
            if metrics['loop_edges'] and metrics['loop_edges'] > 0:
                print("   回环边成功保留在PoseGraph中！")
            if metrics['ate_rmse'] < 0.03:
                print("   达到SOTA水平，ATE < 3cm！")
        elif metrics['loops_detected'] == 0:
            print("⚠️ 回环未检测到，当前数据集可能是平动序列，无回环场景")
        else:
            print("⚠️ ATE较高，请检查位姿图优化配置")
    elif metrics['ate_rmse'] is not None:
        print(f"⚠️ 仅测量到ATE: {metrics['ate_rmse']*100:.2f} cm")

def test_mode():
    print("🧪 运行测试模式（使用真实日志数据）...")
    
    log_content = """
[Tracking] LoopCloser initialized with vocab: ../data/ORBvoc.txt 
 [Tracking] Initialized from frame 0 (1480 features, 1470 map points) 
   [181/200] t= 73.1ms m=  81[DEBUG] New KF created: 6 keyframes total 
 [DEBUG] Candidates: 1 
 [LoopCloser DEBUG] Found 22 valid 2D-3D correspondences 
 [DEBUG] Front-end PoseGraph Addr: 0x599961990450 | Edges BEFORE: 0 
 [DEBUG] Front-end PoseGraph Addr: 0x599961990450 | Edges AFTER: 1 
 
   [LOOP_CLOSED] KF1→KF5 drift=2.3cm 3.3deg 
 [DEBUG] BuildFromKeyframes: kept 1 loop edges 
 [ADDR DEBUG] Back-end PoseGraph Addr: 0x599961990450 | Total edges BEFORE: 6 
 [PGO] Running global pose graph optimization on 6 keyframes... 
 [PGO] Built 6 edges (5 odometry, 1 loop) 
 [PGO] Pre-optimization poses: 
   KF0: pos=(0.0, 0.0, 0.0) 
   KF1: pos=(-0.0, 0.0, 0.0) 
   KF2: pos=(-0.0, 0.0, 0.0) 
   KF3: pos=(-0.0, 0.0, 0.0) 
   KF4: pos=(-0.0, 0.0, 0.0) 
   KF5: pos=(0.0, 0.0, 0.0) 
 [PGO] Fixed 1 keyframe(s) 
 [PGO] Post-optimization poses: 
   KF0: pos=(0.0, 0.0, 0.0) 
   KF1: pos=(-0.0, 0.0, 0.0) 
   KF2: pos=(-0.0, 0.0, 0.0) 
   KF3: pos=(-0.0, 0.0, 0.0) 
   KF4: pos=(-0.0, 0.0, 0.0) 
   KF5: pos=(-0.0, 0.0, 0.0) 
 [PGO] Applied correction: trans=0.0m rot=0.9deg 
   [187/200] t= 60.7ms m= 246[DEBUG] New KF created: 7 keyframes total 
 [DEBUG] Candidates: 2 
 [LoopCloser DEBUG] Found 0 valid 2D-3D correspondences 
 [LoopCloser DEBUG] Not enough 3D points for PnP 
 [DEBUG] BuildFromKeyframes: kept 1 loop edges 
 [ADDR DEBUG] Back-end PoseGraph Addr: 0x599961990450 | Total edges BEFORE: 7 
 [PGO] Running global pose graph optimization on 7 keyframes... 
 [PGO] Built 7 edges (6 odometry, 1 loop) 
 [PGO] Pre-optimization poses: 
   KF0: pos=(0.0, 0.0, 0.0) 
   KF1: pos=(-0.0, 0.0, 0.0) 
   KF2: pos=(-0.0, 0.0, 0.0) 
   KF3: pos=(-0.0, 0.0, 0.0) 
   KF4: pos=(-0.0, 0.0, 0.0) 
   KF5: pos=(-0.0, 0.0, 0.0) 
   KF6: pos=(-0.0, 0.0, -0.0) 
 [PGO] Fixed 1 keyframe(s) 
 [PGO] Post-optimization poses: 
   KF0: pos=(0.0, 0.0, 0.0) 
   KF1: pos=(-0.0, 0.0, 0.0) 
   KF2: pos=(-0.0, 0.0, 0.0) 
   KF3: pos=(-0.0, 0.0, 0.0) 
   KF4: pos=(-0.0, 0.0, 0.0) 
   KF5: pos=(-0.0, 0.0, 0.0) 
   KF6: pos=(-0.0, 0.0, 0.0) 
 [PGO] Applied correction: trans=0.0m rot=53.0deg 
   [188/200] t=194.7ms m= 251[DEBUG] New KF created: 8 keyframes total 
 [DEBUG] Candidates: 3 
 [DEBUG] BuildFromKeyframes: kept 1 loop edges 
 [ADDR DEBUG] Back-end PoseGraph Addr: 0x599961990450 | Total edges BEFORE: 8 
 [PGO] Running global pose graph optimization on 8 keyframes... 
 [PGO] Built 8 edges (7 odometry, 1 loop) 
 [PGO] Pre-optimization poses: 
   KF0: pos=(0.0, 0.0, 0.0) 
   KF1: pos=(-0.02, 0.01, 0.01) 
   KF2: pos=(-0.03, 0.02, 0.03) 
   KF3: pos=(-0.03, 0.01, 0.02) 
   KF4: pos=(-0.02, 0.02, 0.02) 
   KF5: pos=(-0.01, 0.02, 0.03) 
   KF6: pos=(-0.02, 0.01, 0.00) 
   KF7: pos=(-0.01, 0.01, 0.00) 
 [PGO] Fixed 1 keyframe(s) 
 [PGO] Post-optimization poses: 
   KF0: pos=(0.00, 0.00, 0.00) 
   KF1: pos=(-0.02, 0.01, 0.01) 
   KF2: pos=(-0.03, 0.02, 0.03) 
   KF3: pos=(-0.03, 0.01, 0.02) 
   KF4: pos=(-0.02, 0.02, 0.02) 
   KF5: pos=(-0.01, 0.02, 0.03) 
   KF6: pos=(-0.02, 0.01, 0.00) 
   KF7: pos=(-0.01, 0.01, 0.00) 
 [PGO] Applied correction: trans=0.0m rot=0.2deg 
   [200/200] t= 54.8ms m= 251 
 
   Wall time: 16.7s (0.08s/frame) 
 
   [PGO] Running global pose graph optimization... 
 [DEBUG] BuildFromKeyframes: kept 1 loop edges 
   [PGO] Built 8 edges 
 [ADDR DEBUG] Back-end PoseGraph Addr: 0x599961990450 | Total edges BEFORE: 8 
 [PGO] Running global pose graph optimization on 8 keyframes... 
 [PGO] Built 8 edges (7 odometry, 1 loop) 
 [PGO] Pre-optimization poses: 
   KF0: pos=(0.00, 0.00, 0.00) 
   KF1: pos=(-0.02, 0.01, 0.01) 
   KF2: pos=(-0.03, 0.02, 0.03) 
   KF3: pos=(-0.03, 0.01, 0.02) 
   KF4: pos=(-0.02, 0.02, 0.02) 
   KF5: pos=(-0.01, 0.02, 0.03) 
   KF6: pos=(-0.02, 0.01, 0.00) 
   KF7: pos=(-0.01, 0.01, 0.00) 
 [PGO] Fixed 1 keyframe(s) 
 [PGO] Post-optimization poses: 
   KF0: pos=(0.00, 0.00, 0.00) 
   KF1: pos=(-0.02, 0.01, 0.01) 
   KF2: pos=(-0.03, 0.02, 0.03) 
   KF3: pos=(-0.03, 0.01, 0.02) 
   KF4: pos=(-0.02, 0.02, 0.02) 
   KF5: pos=(-0.01, 0.02, 0.03) 
   KF6: pos=(-0.02, 0.01, 0.00) 
   KF7: pos=(-0.01, 0.01, 0.00) 
   [PGO] SUCCESS 
"""
    metrics = parse_log_output(log_content)
    print_metrics(metrics)

def main():
    parser = argparse.ArgumentParser(description='ST-SLAM 4.0 自动运行与指标解析脚本')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    parser.add_argument('--dataset', type=str, default=None, help='数据集路径')
    parser.add_argument('--frames', type=int, default=200, help='运行帧数')
    parser.add_argument('--debug', action='store_true', help='显示完整调试输出')
    parser.add_argument('--output', type=str, default='metrics_report.txt', help='输出文件路径')
    
    args = parser.parse_args()
    
    if args.test:
        test_mode()
        sys.exit(0)
    
    if args.dataset is None:
        args.dataset = '/home/zhang/ORB_SLAM_Learning/datasets/tum/rgbd_dataset_freiburg1_xyz'
        print(f"⚠️ 未指定数据集，使用默认路径: {args.dataset}")
    
    exe_path = '/home/zhang/ORB_SLAM_Learning/st_slam/build/apps/st_slam_main'
    build_dir = '/home/zhang/ORB_SLAM_Learning/st_slam/build'
    
    if not os.path.exists(exe_path):
        print(f"❌ 找不到可执行文件: {exe_path}")
        print("请先编译项目: cd build && make")
        sys.exit(1)
    
    if not os.path.exists(args.dataset):
        print(f"❌ 数据集路径不存在: {args.dataset}")
        sys.exit(1)
    
    cmd = [
        exe_path,
        args.dataset,
        str(args.frames)
    ]
    
    print(f"🚀 正在运行 ST-SLAM 4.0...")
    print(f"📂 可执行文件: {exe_path}")
    print(f"📂 数据集: {args.dataset}")
    print(f"📦 帧数: {args.frames}")
    print("-" * 70)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if args.debug:
            print("\n=== 完整输出 ===")
            print(result.stdout)
            if result.stderr:
                print("=== STDERR ===")
                print(result.stderr)
        else:
            print("\n📝 关键调试信息:")
            print("-" * 45)
            for line in result.stdout.split('\n'):
                if '[DEBUG]' in line or '[PGO]' in line or '[LOOP_CLOSED]' in line or 'ATE' in line:
                    print(line)
            print("-" * 45)
        
        if result.returncode != 0:
            print(f"\n⚠️ 程序返回非零退出码: {result.returncode}")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
        
        metrics = parse_log_output(result.stdout + result.stderr)
        print("\n" + "="*70)
        print_metrics(metrics)
        
        with open(args.output, 'w') as f:
            f.write("ST-SLAM 4.0 Metrics Report\n")
            f.write("="*50 + "\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Frames: {args.frames}\n")
            f.write(f"Timestamp: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
            f.write("="*50 + "\n\n")
            
            f.write("[ATE Metrics]\n")
            f.write(f"ate_rmse_m: {metrics['ate_rmse']}\n")
            f.write(f"ate_mean_m: {metrics['ate_mean']}\n\n")
            
            f.write("[RPE Metrics]\n")
            f.write(f"rpe_trans_rmse_m: {metrics['rpe_trans_rmse']}\n")
            f.write(f"rpe_rot_rmse_deg: {metrics['rpe_rot_rmse']}\n\n")
            
            f.write("[Loop Closure]\n")
            f.write(f"loops_detected: {metrics['loops_detected']}\n")
            f.write(f"loop_edges: {metrics['loop_edges']}\n")
            f.write(f"pgo_correction_m: {metrics['pgo_correction']}\n")
            f.write(f"final_error: {metrics['final_error']}\n")
            f.write(f"iterations: {metrics['iterations']}\n\n")
            
            f.write("[Statistics]\n")
            f.write(f"total_frames: {metrics['total_frames']}\n")
            f.write(f"num_keyframes: {metrics['num_keyframes']}\n")
            f.write(f"edges_built: {metrics['edges_built']}\n")
            f.write(f"tracking_success_rate_pct: {metrics['tracking_success']}\n\n")
            
            f.write("[Performance]\n")
            f.write(f"wall_time_s: {metrics['wall_time']}\n")
        
        print(f"\n📝 指标已保存到 {args.output}")
        
    except subprocess.TimeoutExpired:
        print("❌ 运行超时！")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()