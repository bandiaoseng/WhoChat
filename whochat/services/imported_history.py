"""Explicit archive association and bounded, dated drafting context.

Names discover candidates, never authorize a join. This service has no network
or generation entry point. Imported messages remain historical, not new inbox
events, confirmed current crop facts, or permission to copy/send a suggestion.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from uuid import uuid4
from contextlib import closing

from whochat.services.observed_history import sanitize_excerpt, _QUANTITY

TZ = timezone(timedelta(hours=8))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _owner(conn, scope, contact, conversation=None):
    row = conn.execute('SELECT * FROM contacts WHERE id=? AND account_scope_id=?', (contact,scope)).fetchone()
    if (row is None or row['platform'] != 'wechat' or row['conversation_type'] != 'dm'
            or row['merged_into'] or row['status'] in ('ignored','merged','deleted') or not scope
            or scope == 'account_scope_legacy_unscoped'):
        raise ValueError('archive_customer_identity_invalid')
    if conversation and conn.execute('SELECT 1 FROM conversations WHERE id=? AND contact_id=? AND account_scope_id=?',
                                    (conversation,contact,scope)).fetchone() is None:
        raise ValueError('archive_conversation_identity_invalid')
    return row


def _ticket(row):
    return digest({k: row[k] for k in ('id','account_scope_id','external_contact_id',
        'external_conversation_id','status','contact_id','conversation_id','source_display_name','updated_at')})


def _excerpt(text):
    text = sanitize_excerpt(text)
    return _QUANTITY.sub('[历史数量未核验]', text)[:260] if text else ''


class ImportedHistoryService:
    def __init__(self, db):
        self.db = db

    def profile_archive(self,scope,contact,*,offset=0):
        """Local display only. A history page grants no current-source/AI lease."""
        if type(offset) is not int or not 0<=offset<=1000000:
            raise ValueError('archive_page_invalid')
        with closing(self.db.connect_readonly()) as conn:
            owner=_owner(conn,scope,contact)
            conn.execute('BEGIN')
            join='''FROM historical_chat_messages h
                JOIN historical_contact_mappings m ON m.id=h.mapping_id
                  AND m.account_scope_id=h.account_scope_id AND m.contact_id=h.contact_id
                  AND m.conversation_id=h.conversation_id AND m.external_contact_id=h.external_contact_id
                  AND m.external_conversation_id=h.external_conversation_id
                JOIN conversations c ON c.id=m.conversation_id AND c.account_scope_id=m.account_scope_id
                  AND c.contact_id=m.contact_id
                WHERE m.account_scope_id=? AND m.contact_id=? AND m.status='resolved'
                  AND h.mapping_status='resolved' '''
            stats=conn.execute('SELECT count(*) n,min(h.sent_at) first,max(h.sent_at) last '+join,(scope,contact)).fetchone()
            rows=conn.execute('SELECT h.historical_message_id,h.sent_at,h.sender,h.message_type,h.text '+join+
                ' ORDER BY h.sent_at DESC,h.source_sequence DESC,h.historical_message_id LIMIT 100 OFFSET ?',
                (scope,contact,offset)).fetchall()
            facts=conn.execute('''SELECT count(*) FROM customer_facts f
                JOIN customer_profiles p ON p.id=f.customer_profile_id
                WHERE p.account_scope_id=? AND p.contact_id=?''',
                               (scope,contact)).fetchone()[0]
            return {'name':owner['display_name'],'account_scope_id':scope,'contact_id':contact,
                'total':stats['n'],'first':stats['first'],'last':stats['last'],'offset':offset,
                'structured_fact_count':facts,'records':[dict(r) for r in rows],
                'status':'历史原文关联，不代表已完成全量提炼；旧阶段、旧问题不等于今年现状'}

    def candidates(self, scope, contact):
        with closing(self.db.connect_readonly()) as conn:
            owner = _owner(conn,scope,contact)
            rows = conn.execute('''SELECT * FROM historical_contact_mappings
                WHERE account_scope_id=? AND source_display_name=? AND status='unresolved'
                ORDER BY id LIMIT 8''', (scope,owner['display_name'])).fetchall()
            result = []
            for row in rows:
                stats = conn.execute('''SELECT count(*) n,min(sent_at) first,max(sent_at) last
                    FROM historical_chat_messages WHERE mapping_id=? AND account_scope_id=?''',(row['id'],scope)).fetchone()
                if not stats['n'] or not row['external_contact_id'] or not row['external_conversation_id']:
                    continue
                samples = conn.execute('''SELECT text,sent_at FROM historical_chat_messages
                    WHERE mapping_id=? AND account_scope_id=? AND sender='other' AND message_type='text'
                    ORDER BY sent_at DESC LIMIT 16''',(row['id'],scope)).fetchall()
                result.append({'id':row['id'],'ticket':_ticket(row),'name':row['source_display_name'],
                    'source_identity':row['external_contact_id'], 'count':stats['n'],
                    'first':stats['first'],'last':stats['last'],
                    'samples':[{'date':s['sent_at'],'text':_excerpt(s['text'])} for s in samples if _excerpt(s['text'])][:3]})
            return result

    def confirm(self, *, scope, contact, conversation, mapping_id, ticket, user_confirmed=False):
        if user_confirmed is not True:
            raise ValueError('archive_explicit_confirmation_required')
        with closing(self.db.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            _owner(conn,scope,contact,conversation)
            row = conn.execute('SELECT * FROM historical_contact_mappings WHERE id=?',(mapping_id,)).fetchone()
            if (row is None or row['account_scope_id'] != scope or row['status'] != 'unresolved'
                    or row['contact_id'] or row['conversation_id'] or _ticket(row) != ticket
                    or not row['external_contact_id'] or not row['external_conversation_id']):
                raise ValueError('archive_mapping_changed_or_already_linked')
            # The selected stable source identity may not already represent a
            # different local customer, even in another imported snapshot.
            conflict = conn.execute('''SELECT 1 FROM historical_contact_mappings
                WHERE account_scope_id=? AND external_contact_id=? AND status='resolved'
                AND (contact_id IS NULL OR conversation_id IS NULL OR contact_id<>? OR conversation_id<>?) LIMIT 1''',
                (scope,row['external_contact_id'],contact,conversation)).fetchone()
            if conflict:
                raise ValueError('archive_source_already_belongs_to_another_customer')
            counts = conn.execute('''SELECT count(*) n,sum(CASE WHEN account_scope_id=?
                AND external_contact_id=? AND external_conversation_id=? AND mapping_status='unresolved'
                AND contact_id IS NULL AND conversation_id IS NULL THEN 1 ELSE 0 END) valid
                FROM historical_chat_messages WHERE mapping_id=?''',
                (scope,row['external_contact_id'],row['external_conversation_id'],mapping_id)).fetchone()
            if not counts['n'] or counts['valid'] != counts['n']:
                raise ValueError('archive_rows_do_not_match_selected_identity')
            now = datetime.now(timezone.utc).isoformat()
            conn.execute('''UPDATE historical_contact_mappings SET contact_id=?,conversation_id=?,
                status='resolved',match_basis='explicit_user_current_customer',confidence=1,updated_at=? WHERE id=?''',
                (contact,conversation,now,mapping_id))
            conn.execute('''UPDATE historical_chat_messages SET contact_id=?,conversation_id=?,mapping_status='resolved'
                WHERE mapping_id=?''',(contact,conversation,mapping_id))
            conn.execute('''INSERT INTO app_logs(id,ts,level,module,event,message,context_json)
                VALUES (?,?, 'INFO','imported_history','explicit_mapping_confirmed',?,?)''',
                ('log_'+uuid4().hex,now,'用户确认历史归属；未调用 AI、未创建新消息',
                 json.dumps({'mapping_id':mapping_id,'account_scope_id':scope,'contact_id':contact,
                             'conversation_id':conversation,'rows':counts['n']})))
            return counts['n']


def build_imported_history(conn, view, now=None):
    """Call only after the existing DB/current-view verifier has succeeded."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('archive_timezone_required')
    today = now.astimezone(TZ).date()
    owner = _owner(conn,view.account_scope_id,view.contact_id,view.conversation_id)
    mappings = conn.execute('''SELECT * FROM historical_contact_mappings WHERE account_scope_id=?
        AND contact_id=? AND conversation_id=? AND status='resolved' ORDER BY id LIMIT 8''',
        (view.account_scope_id,view.contact_id,view.conversation_id)).fetchall()
    if not mappings:
        return None
    choices, revisions, seen = [], [], set()
    for m in mappings:
        if not m['external_contact_id'] or not m['external_conversation_id']:
            continue
        revisions.append(_ticket(m))
        query='''SELECT * FROM historical_chat_messages WHERE mapping_id=?
            AND account_scope_id=? AND contact_id=? AND conversation_id=? AND mapping_status='resolved'
            AND external_contact_id=? AND external_conversation_id=? AND message_type='text'
            AND sender IN ('me','other')'''
        params=(m['id'],view.account_scope_id,view.contact_id,m['conversation_id'],
                m['external_contact_id'],m['external_conversation_id'])
        recent=conn.execute(query+' ORDER BY sent_at DESC,source_sequence DESC LIMIT 80',params).fetchall()
        # Query each historical season directly. A busy recent conversation
        # must not push last year's evidence out of a newest-N scan.
        rows=list(recent); recent_ids={r['historical_message_id'] for r in recent}
        for year in range(today.year-5,today.year):
            center=today.replace(year=year,day=min(today.day,28) if today.month==2 else today.day)
            start=(center-timedelta(days=35)).isoformat()
            end=(center+timedelta(days=36)).isoformat()
            rows.extend(conn.execute(query+''' AND substr(sent_at,1,10)>=? AND substr(sent_at,1,10)<?
                ORDER BY sent_at DESC,source_sequence DESC LIMIT 120''',(*params,start,end)).fetchall())
        for row in rows:
            try:
                sent = datetime.fromisoformat(row['sent_at'])
            except (ValueError,TypeError):
                continue
            if sent.tzinfo is None or sent > now:
                continue
            date = sent.astimezone(TZ).date()
            age = (today-date).days
            # Month/day comparison handles year boundaries and leap years;
            # it is a retrieval window, NOT a crop-stage prediction.
            seasonal_day = date.replace(year=2000).timetuple().tm_yday
            now_day = today.replace(year=2000).timetuple().tm_yday
            distance = min(abs(seasonal_day-now_day),366-abs(seasonal_day-now_day))
            same_period = 1 <= today.year-date.year <= 5 and distance <= 35
            if row['historical_message_id'] not in recent_ids and not same_period:
                continue
            text = _excerpt(row['text'])
            key = (row['sender'],row['sent_at'],text)
            if not text or key in seen:
                continue
            seen.add(key)
            score = (100 if same_period else 0)+(30 if row['sender']=='other' else 0)
            score += 15 if any(t in text for t in ('期','种','苗','果','茬','棚','采','效果','病','根')) else 0
            score -= min(age,2000)/2000
            choices.append((score,date,{'date':row['sent_at'],'speaker':row['sender'],
                'text':text,'basis':'prior_year_same_period' if same_period else 'recent_archive',
                'source_id':row['historical_message_id'],
                'source_revision':digest([dict(row),_ticket(m)])}))
    # Reserve recent customer statements; a rich prior-year season must not
    # suppress a current explicit update (the old "already told you my stage"
    # regression). Preserve dates and source roles rather than making facts.
    ranked=sorted(choices,key=lambda x:(x[0],x[1],x[2]['source_id']),reverse=True)
    recent=sorted((x for x in choices if x[2]['speaker']=='other'
                   and 0<=(today-x[1]).days<=30),key=lambda x:x[2]['date'],reverse=True)[:4]
    chosen={x[2]['source_id'] for x in recent}
    selected=recent+[x for x in ranked if x[2]['source_id'] not in chosen][:12-len(recent)]
    excerpts = [x[2] for x in sorted(selected,key=lambda x:(x[1],x[2]['date'],x[2]['source_id']))]
    payload = {'schema':'whochat.imported_history.v1','as_of_local_date':str(today),
        'status':'available' if excerpts else 'no_usable_text','excerpts':excerpts,
        'customer_title_hint':sanitize_excerpt(owner['display_name']),
        'limitations':['bounded_selection_not_full_history','historical_not_current_stage',
                      'title_is_unconfirmed_hint','no_voice_image_or_attachment_content'],
        'mapping_revisions':revisions}
    from whochat.services.activation_season import activation_season_context
    payload['activation_season']=activation_season_context(payload)
    payload['revision'] = digest(payload)
    return payload
