"""Source-bounded calendar hints, never confirmed customer facts or dosing.

Calendar coverage is intentionally explicit. Unknown crops/regions do not get
a guessed nationwide phenology. Historical dates remain separate evidence.
"""
from datetime import date
import re

CALENDAR_VERSION='activation-calendar-20261029-v1'
SOURCE={'id':'HNNY439-2024:table1','title':'湖南省农业技术规程：茄子绿色生产技术规程，表1',
        'publisher':'湖南省农业农村厅','published':'2024-11-26','checked':'2026-10-29',
        'url':'https://www.hnapqs.cn/upload/202412/09/202412091014519290.pdf',
        'scope':'湘南供粤港澳茄子生产；湖南其他区域仅可参照，须确认具体地点和栽培方式'}
# These are coarse calendar windows, not predicted dates for a person.
# Ten-day boundaries are preserved where the source provides them.
_RULES=(
    ('春大棚早熟栽培','育苗',(10,1),(11,30)),
    ('春大棚早熟栽培','定植',(2,11),(2,29)),
    ('春大棚早熟栽培','采收',(4,11),(8,31)),
    ('春夏露地栽培','育苗',(11,1),(12,31)),
    ('春夏露地栽培','定植',(4,1),(4,10)),
    ('春夏露地栽培','采收',(5,11),(8,31)),
    ('大棚秋延后栽培','育苗',(6,21),(7,10)),
    ('大棚秋延后栽培','定植',(7,21),(8,31)),
    ('大棚秋延后栽培','采收',(9,1),(11,30)),
    ('高山茄子栽培','育苗',(12,1),(2,29)),
    ('高山茄子栽培','定植',(5,11),(5,20)),
    ('高山茄子栽培','采收',(6,11),(10,31)),
)


def activation_season_context(history):
    today=date.fromisoformat(history['as_of_local_date'])
    title=history.get('customer_title_hint','')
    crops=sorted(set(re.findall(r'茄子|辣椒|番茄|草莓|葡萄|蓝莓|柑橘|柑桔|柚子|西瓜|玉米|大豆|水稻|桃',title)))
    regions=sorted(set(re.findall(r'湖南|湖北|河南|河北|山东|山西|陕西|四川|云南|浙江|江苏|上海|广东|广西|海南|辽宁|吉林|黑龙江|安徽|福建|江西|贵州|甘肃|青海|宁夏|新疆|内蒙古|重庆|北京|天津|西藏',title)))
    result={'schema':'whochat.activation_season.v1','version':CALENDAR_VERSION,
        'as_of_local_date':str(today),'crop_hints':crops,'region_hints':regions,
        'hint_source':'customer_title_unconfirmed','current_stage_confirmed':False,
        'prior_year_same_period_sources':[e['source_id'] for e in history.get('excerpts',())
            if e.get('basis')=='prior_year_same_period' and e.get('speaker')=='other'],
        'calendar_status':'unsupported_or_ambiguous','possible_windows':[],
        'use':'仅用于激活时询问近况；近期明确客户记录优先，不回写客户事实、不推断病害、不提供用量'}
    if crops!=['茄子'] or regions!=['湖南']:
        return result
    md=(today.month,today.day)
    for mode,stage,start,end in _RULES:
        active=start<=md<=end if start<=end else md>=start or md<=end
        if active:
            result['possible_windows'].append({'cultivation_mode':mode,'possible_stage':stage,
                'condition':'今年仍种该作物，且该地块采用此栽培方式；目前均未确认',
                'source_id':SOURCE['id']})
    result['calendar_status']='reference_only' if result['possible_windows'] else 'no_matching_window'
    result['source']=dict(SOURCE)
    return result
