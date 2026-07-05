---
name: pr-description-writer
description: PR 描述生成:根据 diff 和修改记录写清晰的 pull request 说明
---

# PR 描述生成流程

1. 用 `git_diff` 拿到完整改动
2. 按模板组织:

```markdown
## 改了什么
一句话概括 + 按文件列出关键改动

## 为什么
关联的 issue / bug 现象

## 怎么验证
跑了哪些测试、结果如何
```

原则:
- 描述面向 reviewer:先结论后细节,diff 里能看到的行级细节不复述
- 有行为变化必须显式标注(尤其是接口签名、默认值、返回格式)
