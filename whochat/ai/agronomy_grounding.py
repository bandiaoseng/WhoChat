"""Bounded, local premise checks. Not a general agronomic truth verifier.

No IO, model call, customer mutation, learning update, or copy authorization.
Input provenance remains with the original context. A seller's past diagnosis
and a bare customer acknowledgement cannot corroborate that diagnosis.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


GROUNDING_POLICY = "agro-premise-v1-20261030"
# Reviewed descriptions, not crop treatments or customer-specific facts.
FACILITY_REFERENCES = (
    ("承德市农业农村局·葡萄设施栽培技术", "https://nyncj.chengde.gov.cn/art/2022/2/25/art_667_834239.html",
     "介绍仅在棚顶部覆膜的避雨栽培；该说明不能证明某客户实际四周是否封闭。"),
    ("四川省农业农村厅·2022年度农业主推技术·蔬菜避雨栽培", "https://nynct.sc.gov.cn/nynct/c100664/2021/12/15/63b366b5be1147bd908c583cf82dfb6b/files/543df9110eb64a9f9712ffc0cbb928ad.pdf",
     "同时介绍多功能棚和小拱棚等形式，部分有卷膜通风设施；不能将一种棚型套用到所有避雨棚。"),
)


def grounding_instruction() -> str:
    return (
        "农艺事实与推测边界：先核对证据，再写话术。客户明确陈述是客户自述，不等于已确诊；"
        "我方历史分析、以前AI草稿、产品宣传和人工改写均不自动成为已核实的田间事实。"
        "同一句话被复制发出、重复出现，不会增加真实性。客户‘对/是/嗯/好’只承接相邻问题，"
        "不能把我方整段病因、棚型、湿度或用肥判断全部确认。"
        "客户备注/昵称不是当前作物证据；买过不等于用过，地区和月份不等于确定生育期。"
        "先承接明确已知情况，不重复泛问已确认资料；分清当前、历史、不同地块及有冲突的信息。"
        "没有现场证据的环境或原因只能写为待核实假设，不把假设放进当前判断的确定前提。"
        "例如‘通风差可能影响授粉’仍然默认通风差，不能用后面的‘可能’掩盖前提缺证据。"
        "追问也要检查前提：未确认有通风口，不问通风口怎么开；未确认使用，不问使用后效果。"
        "只问会实质影响判断的一个关键缺口，不机械要求客户填写整套资料。"
        "回答前复核客户摘要、当前判断、追问和两条话术是否一致：确定性客户结论必须有对应证据；"
        "证据不足时用中性的核实问题或带明确条件的可能性，不编造农艺结论，不自动沿用我方旧判断。"
    )


def grounding_context(context) -> str:
    """Compact guidance alongside the existing role-labelled chat; no new data."""
    messages = list(context.effective_messages)[-12:]
    own = sum(getattr(m.speaker, "value", m.speaker) == "me" for m in messages)
    other = sum(getattr(m.speaker, "value", m.speaker) == "other" for m in messages)
    lines = [f"本次可见原话：客户{other}条、我方{own}条。以下聊天保留发送方；不是每一句都属于客户事实。",
             "我方的分析只能说明以前怎样沟通过。若与客户陈述或可靠知识不符，应纠正或核实，不为保持前后口径而强化旧错误。",
             "简短同意不能独立确认病因；既不能忽略明确回答，也不能扩大其确认范围。"]
    if any(re.search(r"避雨|遮雨", m.text) for m in messages):
        lines.append("棚型边界：避雨是用途，不等于封闭温室，也不保证全部四周敞开。"
                     "仅凭‘避雨棚’不能认定通风差、湿度高、存在可开关风口或授粉异常；"
                     "未确认实际结构时不要按封闭棚给开风口建议。四周敞开也不等于任何时候都没有微环境问题。")
        lines.extend(f"参考：{title}（{url}）：{description}" for title, url, description in FACILITY_REFERENCES)
    return "\n".join(lines)


@dataclass(frozen=True)
class PremiseRule:
    code: str
    pattern: re.Pattern
    label: str


# These are finite linguistic checks, deliberately not automatic diagnoses.
# Every rule uses the same provenance/qualification path, not per-customer exceptions.
ENVIRONMENT_RULES = (
    PremiseRule("ventilation", re.compile(r"(?:通风|透气)(?:性|条件)?(?:比较|较|偏|不太|很|太|特别|非常)?(?:差|不好|不足|不良|不畅)"), "通风不良"),
    PremiseRule("humidity", re.compile(r"(?:湿度|空气湿度)(?:比较|较|偏|很|太|特别|非常|过)?高|(?:棚内|棚里|环境)(?:比较|很|太|特别|非常)?潮湿"), "湿度偏高"),
    PremiseRule("temperature", re.compile(r"(?:棚内温度|棚里温度|温度)(?:比较|较|偏|很|太|特别|非常|过)高"), "温度偏高"),
    PremiseRule("enclosure", re.compile(r"(?:四周|周围|两侧)(?:都|是|全部)?(?:封闭|封住|围起来)|密闭棚|封闭式大棚"), "封闭结构"),
)
EQUIPMENT = re.compile(r"通风口|放风口|顶风口|侧风口|卷膜器|侧膜|天窗")
OPERATION = re.compile(r"怎么|如何|开|关|调|掀|卷|放风")
CLAUSES = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
NEGATION = re.compile(r"(?:不一定|并非|未必|不能认定|不能说|不能确定|不能断定|不是|并不|没有|不算|不属于|未确认|尚未确认|未知|不确定|是否|有没有|会不会|是不是|不能仅凭|不代表)[^，,。；;]{0,18}$")
HYPOTHESIS = re.compile(r"(?:如果|假如|若是|假设|可能|也许|或许|怀疑|疑似)[^。；;]{0,45}$")
REPORTED = re.compile(r"(?:你说|您说|你之前说|您之前说|听说|别人说|网上说|AI说|AI认为|去年|往年|以前|之前|曾经|另一块|别的棚)", re.I)


def _qualified(clause: str, start: int, end: int) -> bool:
    # A tentative word in a different comma-clause must not launder a later
    # definite premise ("可能缺肥，棚里湿度高" is still an unsupported humidity claim).
    prefix = re.split(r"[，,、]", clause[:start])[-1]
    suffix = clause[end:]
    return bool(NEGATION.search(prefix) or HYPOTHESIS.search(prefix)
                or re.match(r"(?:不差|不高|吗|么|是否|与否|时|的情况下|的话)", suffix)
                or re.match(r"(?:的判断|这个判断|的说法)?(?:不准确|不严谨|未经确认|尚未核实)", suffix)
                or re.match(r"[？?]", suffix))


def _customer_reports(context, pattern: re.Pattern) -> bool:
    """Only explicit direct customer text. A later denial revokes support.

    Neither seller messages, nickname, learned feedback, old imported material,
    nor a generic knowledge document can establish a current site condition.
    This checks reported evidence only, never claims it is a measurement.
    """
    supported = False
    for message in list(context.effective_messages)[-12:]:
        if getattr(message.speaker, "value", message.speaker) != "other":
            continue
        for clause in CLAUSES.findall(message.text):
            for match in pattern.finditer(clause):
                if REPORTED.search(clause[:match.start()]):
                    continue
                supported = not _qualified(clause, match.start(), match.end())
    return supported


def _result_texts(result):
    return (result.customer_summary, result.problem_assessment, result.ask_first,
            result.sales_judgement, result.next_action, *result.uncertainties,
            *(s.text for s in result.suggestions),
            *(s.rationale for s in result.suggestions),
            *(value for item in result.ingredient_advice
              for value in (item.role, item.rationale, *item.applicable_conditions)))


def premise_error(result, context) -> str | None:
    """Return reason codes only. No repair-by-deletion or hidden model retry."""
    reports = {rule.code: _customer_reports(context, rule.pattern) for rule in ENVIRONMENT_RULES}
    equipment = {name: _customer_reports(context, re.compile(re.escape(name)))
                 for name in set(EQUIPMENT.findall("\n".join(_result_texts(result))))}
    for text in _result_texts(result):
        for clause in CLAUSES.findall(text):
            for rule in ENVIRONMENT_RULES:
                for match in rule.pattern.finditer(clause):
                    if not reports[rule.code] and not _qualified(clause, match.start(), match.end()):
                        return "agronomy_premise_unsupported_" + rule.code
            for match in EQUIPMENT.finditer(clause):
                if (not equipment[match.group()] and OPERATION.search(clause)
                        and not _qualified(clause, match.start(), match.end())):
                    return "agronomy_premise_unconfirmed_equipment"
            # Definitive causal diagnosis is not licensed by a symptom or an
            # operator's earlier assertion. Tentative, conditional explanations remain allowed.
            for match in re.finditer(r"(?:就是|肯定是|一定是|确定是|说明是)(?:因为)?(?:缺[氮磷钾钙镁硼锌铁]|药害|肥害|授粉不良|授粉不足|授粉问题)", clause):
                prefix = re.split(r"[，,、]", clause[:match.start()])[-1]
                if not (NEGATION.search(prefix) or re.search(r"(?:不能|无法|未能|尚不能)$", prefix)):
                    return "agronomy_premise_definitive_cause"
    return None


def premise_failure_message(reason: str) -> str:
    labels = {"ventilation": "把未经确认的通风状况作为前提",
              "humidity": "把未经确认的湿度状况作为前提",
              "temperature": "把未经确认的温度状况作为前提",
              "enclosure": "默认了尚未确认的棚体结构",
              "unconfirmed_equipment": "默认存在尚未确认的风口或设施",
              "definitive_cause": "将缺少充分依据的原因写成确定诊断"}
    for code, label in labels.items():
        if "agronomy_premise_" + ("unsupported_" if code in {r.code for r in ENVIRONMENT_RULES} else "") + code in reason:
            return "AI 草稿" + label + "，已拦截，未提供可复制话术。可补充现场信息后手动生成；本次实际调用已计次，未自动重试"
    return ""
