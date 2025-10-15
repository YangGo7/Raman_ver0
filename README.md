<<<<<<< HEAD
# 치과 AI 진단 시스템 - 아키텍처 설계

## 🎯 프로젝트 개요

파노라마 X-ray 이미지를 분석하여 치아 상태를 진단하고 AI 리포트를 생성하는 시스템

### 핵심 목표
- ⚡ 실시간 분석 (< 200ms)
- 🎯 높은 정확도 (> 85% mAP)
- 🔧 유지보수 용이성
- 📈 확장 가능성

---

## 🏗️ 시스템 아키텍처

### 3-Tier Model Architecture

#### **TIER 1: Foundation**
```
Model: tooth_seg.pt (YOLOv8-seg)
Purpose: 치아 instance segmentation + numbering
Output: 32 tooth instances
Inference: ~50ms
Status: ✅ 완료
```

#### **TIER 2: Clinical**
```
Model: periodontal.pt (YOLOv8-seg)
Purpose: 치주 상태 평가
Classes:
  - class 0: bone_level
  - class 1: cej
  - class 2: calculus (future)
Inference: ~40ms
Status: 🔄 통합 필요
```

#### **TIER 3: Pathology**
```
Model: findings.pt (YOLOv8-seg, Multi-class)
Purpose: 병변 및 이상 소견
Classes:
  - class 0: caries
  - class 1: periapical_lesion
  - class 2: crown
  - class 3: rct
  - class 4: fracture
  - class 5: implant
  - class 6: missing_tooth
  - class 7: impacted_tooth
Inference: ~50ms
Status: 🆕 신규 개발
```

---

## 📊 데이터 플로우

1. **입력**: 파노라마 X-ray (2048x2048)
2. **전처리**: Resize/Normalize (10ms)
3. **Foundation**: Tooth segmentation (50ms)
4. **Parallel Inference**:
   - Clinical assessment (40ms)
   - Pathology detection (50ms)
5. **Integration**: Spatial fusion + validation (15ms)
6. **Reasoning**: Clinical analysis
7. **LLM**: Report generation (2000ms, async)
8. **Output**: Interactive UI + PDF report

**Total**: ~140ms (vision), ~3s (with LLM)

---

## 🎨 시각화 레이어

### Color Scheme
- **Tooth boundaries**: White outline (α=0.8)
- **Bone level**: Blue #3498DB (α=0.5)
- **CEJ**: Cyan #1ABC9C (α=0.5)
- **Caries**: Red #E74C3C (α=0.6)
- **Periapical lesion**: Orange #E67E22
- **Crown**: Gold #F39C12 (α=0.5)
- **RCT**: Purple #9B59B6 (α=0.5)
- **Fracture**: Dark Red #C0392B
- **Implant**: Silver #95A5A6 (α=0.6)
- **Calculus**: Lime #9CFF2E (α=0.6)

### Smart Presets
- **Overview**: Tooth + Critical findings
- **Periodontal**: Tooth + Bone + CEJ + Calculus
- **Treatment Plan**: All layers
- **Patient View**: Simplified

---

## 🚀 구현 로드맵

### Phase 1: Core MVP (2개월)
- [x] Week 1-2: Findings 모델 통합
- [ ] Week 3-4: Periodontal 모델 통합
- [ ] Week 5-6: Streamlit UI v1
- [ ] Week 7-8: LLM Integration

**Deliverable**: 동작하는 3-tier 시스템

### Phase 2: Enhancement (2개월)
- [ ] Week 9-10: 추가 Detection Classes
- [ ] Week 11-12: Advanced Visualization
- [ ] Week 13-14: Clinical Reasoning Engine
- [ ] Week 15-16: Polish & Testing

**Deliverable**: Production-ready

### Phase 3: Advanced (2-3개월)
- [ ] Week 17-20: Rare Findings
- [ ] Week 21-24: Intelligence Layer
- [ ] Week 25-28: Enterprise Features

**Deliverable**: Enterprise-grade

---

## 📈 성능 목표

### Accuracy Targets
| Task | mAP@0.5 | Recall |
|------|---------|--------|
| Tooth Seg | 0.95 | 0.93 |
| Bone Level | 0.90 | 0.88 |
| Caries | 0.88 | 0.85 |
| Lesion | 0.86 | 0.82 |
| Crown | 0.94 | 0.92 |

### Speed Targets
- Vision Pipeline: < 200ms ✅
- Total (with LLM): < 3s ✅
- User Experience: Excellent

---

## 💾 출력 데이터 구조
```json
{
  "teeth": {
    "14": {
      "present": true,
      "findings": [
        {
          "type": "caries",
          "mask": [[x,y], ...],
          "confidence": 0.94,
          "severity": "moderate"
        }
      ],
      "bone_level": 72.5,
      "priority": "high"
    }
  },
  "summary": {
    "total_teeth": 28,
    "findings_count": {...},
    "priority_teeth": ["14", "21"]
  },
  "ai_report": "..."
}
```

---

## 🔧 기술 스택

### Core
- **Framework**: YOLOv8
- **Language**: Python 3.9+
- **UI**: Streamlit
- **LLM**: GPT-4 / Claude

### Infrastructure
- **GPU**: RTX 3090 / A10
- **Storage**: 500GB SSD
- **RAM**: 32GB+

### Tools
- **Design**: Figma, Excalidraw
- **Docs**: Notion
- **PM**: Linear
- **Version Control**: Git

---

## 📚 참고 자료

### 다이어그램
- 전체 아키텍처: [Excalidraw link]
- 데이터 플로우: [Mermaid diagram]
- UI Mockup: [Figma link]

### 문서
- API Reference: [Link]
- User Guide: [Link]
- Development Guide: [Link]

---

## 👥 팀 & 역할

- **ML Engineer**: 모델 개발 및 학습
- **Backend**: 파이프라인 통합
- **Frontend**: UI/UX 개발
- **PM**: 프로젝트 관리

---

## 📞 Contact

- GitHub: [Repository link]
- Documentation: [Notion link]
- Issues: [Linear workspace]

---

*Last updated: 2025-01-15*
=======
# Raman_ver0 
>>>>>>> cf2dbf9 (Add preprocessing pipeline and configuration)
