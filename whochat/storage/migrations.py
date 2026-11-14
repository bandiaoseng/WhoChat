from __future__ import annotations


MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE strategies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            goal TEXT NOT NULL,
            mode TEXT NOT NULL,
            tone TEXT NOT NULL,
            avoid TEXT NOT NULL DEFAULT '',
            reply_variants TEXT NOT NULL DEFAULT '',
            requires_manual_reply INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE contacts (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            display_name TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            status TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            remark TEXT NOT NULL DEFAULT '',
            avatar_fingerprint TEXT NOT NULL DEFAULT '',
            merged_into TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(strategy_id) REFERENCES strategies(id),
            FOREIGN KEY(merged_into) REFERENCES contacts(id)
        );

        CREATE TABLE contact_aliases (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(contact_id, alias),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );

        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            speaker TEXT NOT NULL,
            text TEXT NOT NULL,
            content_type TEXT NOT NULL,
            ocr_confidence REAL,
            observed_at TEXT NOT NULL,
            message_time TEXT,
            time_source TEXT NOT NULL,
            partial INTEGER NOT NULL DEFAULT 0,
            fingerprint TEXT NOT NULL,
            source TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_messages_contact_observed ON messages(contact_id, observed_at);
        CREATE UNIQUE INDEX idx_messages_fingerprint ON messages(contact_id, fingerprint);

        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL,
            source_message_id TEXT,
            expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(source_message_id) REFERENCES messages(id)
        );
        CREATE INDEX idx_memories_contact_status ON memories(contact_id, status);

        CREATE TABLE app_logs (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            level TEXT NOT NULL,
            module TEXT NOT NULL,
            event TEXT NOT NULL,
            message TEXT NOT NULL,
            context_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_app_logs_ts ON app_logs(ts);
        """,
    )
    ,
    (
        2,
        """
        CREATE TABLE layout_calibrations (
            id TEXT PRIMARY KEY,
            target TEXT NOT NULL,
            name TEXT NOT NULL,
            theme TEXT NOT NULL,
            dpi_scale REAL NOT NULL DEFAULT 1.0,
            nav_rect_json TEXT NOT NULL,
            chat_list_rect_json TEXT NOT NULL,
            content_rect_json TEXT NOT NULL,
            title_rect_json TEXT NOT NULL,
            message_rect_json TEXT NOT NULL,
            input_rect_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX idx_layout_calibrations_active
        ON layout_calibrations(target, active)
        WHERE active = 1;
        """,
    ),
    (
        3,
        """
        CREATE TABLE generation_logs (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            contact_id TEXT,
            strategy_id TEXT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            allowed INTEGER NOT NULL,
            status TEXT NOT NULL,
            suggestion_count INTEGER NOT NULL,
            risk_summary TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            page_type TEXT NOT NULL,
            page_confidence REAL NOT NULL,
            message_count INTEGER NOT NULL,
            memory_count INTEGER NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_generation_logs_ts ON generation_logs(ts);
        CREATE INDEX idx_generation_logs_contact_ts ON generation_logs(contact_id, ts);
        CREATE INDEX idx_generation_logs_context ON generation_logs(context_hash);
        """,
    ),
    (
        4,
        """
        ALTER TABLE contacts ADD COLUMN allow_cloud_ai INTEGER NOT NULL DEFAULT 0;
        """,
    ),
    (
        5,
        """
        CREATE TABLE people (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL,
            remark TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE person_aliases (
            id TEXT PRIMARY KEY,
            person_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(person_id, alias),
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_person_aliases_alias ON person_aliases(alias);

        CREATE TABLE contact_person_links (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            person_id TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            verified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(contact_id, person_id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_contact_person_links_contact ON contact_person_links(contact_id);
        CREATE INDEX idx_contact_person_links_person ON contact_person_links(person_id);

        CREATE TABLE group_members (
            id TEXT PRIMARY KEY,
            group_contact_id TEXT NOT NULL,
            member_display_name TEXT NOT NULL,
            person_id TEXT,
            platform_contact_id TEXT,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(group_contact_id, member_display_name),
            FOREIGN KEY(group_contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(person_id) REFERENCES people(id) ON DELETE SET NULL,
            FOREIGN KEY(platform_contact_id) REFERENCES contacts(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_group_members_group ON group_members(group_contact_id);
        CREATE INDEX idx_group_members_person ON group_members(person_id);
        """,
    ),
    (
        6,
        """
        CREATE TABLE settings_audit (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            actor TEXT NOT NULL,
            scope TEXT NOT NULL,
            changes_json TEXT NOT NULL,
            secret_backend TEXT NOT NULL
        );
        CREATE INDEX idx_settings_audit_ts ON settings_audit(ts);
        CREATE INDEX idx_settings_audit_scope_ts ON settings_audit(scope, ts);
        """,
    ),
    (
        7,
        """
        ALTER TABLE strategies ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX idx_strategies_archived_name ON strategies(archived, name);
        """,
    ),
    (
        8,
        """
        CREATE TABLE capture_samples (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            job_id INTEGER NOT NULL,
            hwnd INTEGER,
            snapshot_hash TEXT NOT NULL,
            image_path TEXT NOT NULL,
            ocr_image_path TEXT NOT NULL,
            crop_rect_json TEXT NOT NULL,
            ocr_engine TEXT NOT NULL,
            ocr_warning TEXT NOT NULL,
            page_type TEXT NOT NULL,
            page_confidence REAL NOT NULL,
            message_count INTEGER NOT NULL,
            retained_image INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_capture_samples_ts ON capture_samples(ts);
        CREATE INDEX idx_capture_samples_hash ON capture_samples(snapshot_hash);
        """,
    ),
    (
        9,
        """
        ALTER TABLE capture_samples ADD COLUMN target_app TEXT NOT NULL DEFAULT 'wechat';
        ALTER TABLE capture_samples ADD COLUMN app_label TEXT NOT NULL DEFAULT '微信';
        CREATE INDEX idx_capture_samples_target_ts ON capture_samples(target_app, ts);
        """,
    ),
    (
        10,
        """
        ALTER TABLE capture_samples ADD COLUMN title_ocr_image_path TEXT NOT NULL DEFAULT '';
        ALTER TABLE capture_samples ADD COLUMN title_crop_rect_json TEXT NOT NULL DEFAULT '';
        ALTER TABLE capture_samples ADD COLUMN title_ocr_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN content_ocr_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN total_elapsed_ms INTEGER;
        """,
    ),
    (
        11,
        """
        CREATE TABLE reply_feedback (
            id TEXT PRIMARY KEY,
            ts TEXT NOT NULL,
            contact_id TEXT,
            strategy_id TEXT,
            provider TEXT NOT NULL,
            status TEXT NOT NULL,
            suggestion_label TEXT NOT NULL,
            suggestion_text_preview TEXT NOT NULL,
            risk TEXT NOT NULL,
            feedback TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            page_type TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            memory_count INTEGER NOT NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(strategy_id) REFERENCES strategies(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_reply_feedback_ts ON reply_feedback(ts);
        CREATE INDEX idx_reply_feedback_contact_ts ON reply_feedback(contact_id, ts);
        CREATE INDEX idx_reply_feedback_context ON reply_feedback(context_hash);
        """,
    ),
    (
        12,
        """
        ALTER TABLE messages ADD COLUMN sender_name TEXT NOT NULL DEFAULT '';
        CREATE INDEX idx_messages_contact_sender ON messages(contact_id, sender_name);
        """,
    ),
    (
        13,
        """
        UPDATE contacts SET allow_cloud_ai = 1 WHERE allow_cloud_ai = 0;
        """,
    ),
    (
        14,
        """
        ALTER TABLE memories ADD COLUMN fingerprint TEXT NOT NULL DEFAULT '';
        UPDATE memories SET fingerprint = 'legacy:' || id WHERE fingerprint = '';
        CREATE UNIQUE INDEX idx_memories_fingerprint ON memories(contact_id, fingerprint);
        """,
    ),
    (
        15,
        """
        -- Migration 13 granted cloud AI to every contact unconditionally, so an
        -- OCR-discovered contact could have its chat content sent to a third-party
        -- provider without the user ever agreeing.  Cloud sending is now explicit
        -- opt-in per contact (contacts page "AI开/AI关" switch, or the checkbox in
        -- the contact edit dialog), so the blanket grant is reverted.  Users who
        -- want cloud replies re-authorise the contacts they care about once.
        UPDATE contacts SET allow_cloud_ai = 0 WHERE allow_cloud_ai = 1;
        """,
    ),
    (
        16,
        """
        -- Phase 1: stable identity bindings and a restart-safe conversation
        -- timeline. Existing OCR messages are preserved and attached to a
        -- deterministic legacy conversation/session/segment per contact.
        CREATE TABLE contact_identity_bindings (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            avatar_fingerprint TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0,
            verified INTEGER NOT NULL DEFAULT 0,
            last_confirmed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(platform, conversation_type, normalized_title),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_identity_bindings_contact ON contact_identity_bindings(contact_id);

        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            contact_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            identity_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(contact_id, platform, conversation_type),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_conversations_identity ON conversations(identity_key);

        CREATE TABLE conversation_sessions (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            hwnd INTEGER,
            stable_title TEXT NOT NULL,
            identity_confidence REAL NOT NULL,
            started_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_conversation_sessions_timeline ON conversation_sessions(conversation_id, started_at);

        CREATE TABLE conversation_segments (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            page_type TEXT NOT NULL,
            page_confidence REAL NOT NULL,
            ocr_engine TEXT NOT NULL,
            reliable INTEGER NOT NULL DEFAULT 1,
            order_uncertain INTEGER NOT NULL DEFAULT 0,
            source_job_id INTEGER,
            message_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(conversation_id, snapshot_hash),
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY(session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_conversation_segments_timeline ON conversation_segments(conversation_id, observed_at);

        ALTER TABLE messages ADD COLUMN conversation_id TEXT;
        ALTER TABLE messages ADD COLUMN session_id TEXT;
        ALTER TABLE messages ADD COLUMN segment_id TEXT;
        ALTER TABLE messages ADD COLUMN sequence_no INTEGER;
        ALTER TABLE messages ADD COLUMN segment_index INTEGER;
        ALTER TABLE messages ADD COLUMN bbox_json TEXT NOT NULL DEFAULT '';
        ALTER TABLE messages ADD COLUMN order_uncertain INTEGER NOT NULL DEFAULT 0;

        INSERT INTO conversations(id, contact_id, platform, conversation_type, identity_key, created_at, updated_at)
        SELECT 'conversation_legacy_' || id, id, platform, conversation_type,
               platform || '|' || conversation_type || '|' || lower(trim(display_name)), created_at, updated_at
        FROM contacts;

        INSERT INTO conversation_sessions(id, conversation_id, hwnd, stable_title, identity_confidence,
                                          started_at, last_seen_at, ended_at, source)
        SELECT 'session_legacy_' || c.id, 'conversation_legacy_' || c.id, NULL, c.display_name, 0.5,
               COALESCE(MIN(m.observed_at), c.created_at), COALESCE(MAX(m.observed_at), c.updated_at),
               COALESCE(MAX(m.observed_at), c.updated_at), 'migration_v16'
        FROM contacts c LEFT JOIN messages m ON m.contact_id = c.id
        GROUP BY c.id;

        INSERT INTO conversation_segments(id, conversation_id, session_id, snapshot_hash, observed_at,
                                          page_type, page_confidence, ocr_engine, reliable,
                                          order_uncertain, source_job_id, message_count)
        SELECT 'segment_legacy_' || c.id, 'conversation_legacy_' || c.id, 'session_legacy_' || c.id,
               'legacy:' || c.id, COALESCE(MIN(m.observed_at), c.created_at),
               CASE WHEN c.conversation_type = 'group' THEN 'chat_group' ELSE 'chat_dm' END,
               0.5, 'legacy', 0, 1, NULL, COUNT(m.id)
        FROM contacts c LEFT JOIN messages m ON m.contact_id = c.id
        GROUP BY c.id;

        UPDATE messages
        SET conversation_id = 'conversation_legacy_' || contact_id,
            session_id = 'session_legacy_' || contact_id,
            segment_id = 'segment_legacy_' || contact_id,
            sequence_no = (
                SELECT COUNT(*) FROM messages older
                WHERE older.contact_id = messages.contact_id
                  AND (older.observed_at < messages.observed_at
                       OR (older.observed_at = messages.observed_at AND older.id <= messages.id))
            ),
            segment_index = (
                SELECT COUNT(*) - 1 FROM messages older
                WHERE older.contact_id = messages.contact_id
                  AND (older.observed_at < messages.observed_at
                       OR (older.observed_at = messages.observed_at AND older.id <= messages.id))
            ),
            order_uncertain = 1;
        CREATE INDEX idx_messages_conversation_sequence ON messages(conversation_id, sequence_no);
        CREATE INDEX idx_messages_segment ON messages(segment_id, segment_index);

        CREATE TABLE ai_analysis_events (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL UNIQUE,
            contact_id TEXT NOT NULL,
            session_id TEXT,
            segment_id TEXT,
            status TEXT NOT NULL,
            context_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_ai_analysis_events_contact ON ai_analysis_events(contact_id, created_at);
        """,
    ),
    (
        17,
        """
        -- P0 account boundary + observation quarantine.  All v16 data is
        -- preserved under an explicit legacy scope; newly observed windows
        -- cannot match those global title-only bindings.
        CREATE TABLE account_scopes (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            local_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(platform, local_name)
        );

        INSERT INTO account_scopes(id, platform, local_name, status, created_at, updated_at)
        VALUES ('account_scope_legacy_unscoped', 'wechat', '历史未分账号数据', 'legacy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

        CREATE TABLE target_instances (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            account_scope_id TEXT,
            runtime_key TEXT NOT NULL,
            visual_hash TEXT NOT NULL DEFAULT '',
            visual_reliable INTEGER NOT NULL DEFAULT 0,
            process_name TEXT NOT NULL DEFAULT '',
            executable_path TEXT NOT NULL DEFAULT '',
            last_pid INTEGER,
            last_hwnd INTEGER,
            last_rect_json TEXT NOT NULL DEFAULT '',
            binding_source TEXT NOT NULL DEFAULT 'unbound',
            confidence REAL NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(runtime_key, visual_hash),
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_target_instances_scope ON target_instances(account_scope_id, last_seen_at);
        CREATE INDEX idx_target_instances_visual ON target_instances(platform, visual_hash);

        ALTER TABLE contacts ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        CREATE INDEX idx_contacts_scope_name ON contacts(account_scope_id, platform, display_name);

        DROP INDEX idx_identity_bindings_contact;
        ALTER TABLE contact_identity_bindings RENAME TO contact_identity_bindings_v16;
        CREATE TABLE contact_identity_bindings (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            platform TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            avatar_fingerprint TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0,
            verified INTEGER NOT NULL DEFAULT 0,
            last_confirmed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(account_scope_id, platform, conversation_type, normalized_title),
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        INSERT INTO contact_identity_bindings(
            id, account_scope_id, platform, conversation_type, normalized_title, contact_id,
            avatar_fingerprint, confidence, verified, last_confirmed_at, created_at, updated_at
        )
        SELECT id, 'account_scope_legacy_unscoped', platform, conversation_type, normalized_title, contact_id,
               avatar_fingerprint, confidence, verified, last_confirmed_at, created_at, updated_at
        FROM contact_identity_bindings_v16;
        DROP TABLE contact_identity_bindings_v16;
        CREATE INDEX idx_identity_bindings_contact ON contact_identity_bindings(contact_id);
        CREATE INDEX idx_identity_bindings_scope ON contact_identity_bindings(account_scope_id, normalized_title);

        ALTER TABLE conversations ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        CREATE INDEX idx_conversations_scope_identity ON conversations(account_scope_id, identity_key);

        ALTER TABLE conversation_sessions ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        ALTER TABLE conversation_sessions ADD COLUMN target_instance_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE conversation_sessions ADD COLUMN capture_generation INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX idx_conversation_sessions_scope ON conversation_sessions(account_scope_id, conversation_id, started_at);

        ALTER TABLE conversation_segments ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        ALTER TABLE conversation_segments ADD COLUMN target_instance_id TEXT NOT NULL DEFAULT '';
        CREATE INDEX idx_conversation_segments_scope ON conversation_segments(account_scope_id, conversation_id, observed_at);

        ALTER TABLE messages ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        CREATE INDEX idx_messages_scope_conversation ON messages(account_scope_id, conversation_id, sequence_no);

        ALTER TABLE ai_analysis_events ADD COLUMN account_scope_id TEXT NOT NULL DEFAULT 'account_scope_legacy_unscoped';
        CREATE INDEX idx_ai_analysis_events_scope ON ai_analysis_events(account_scope_id, contact_id, created_at);

        CREATE TABLE conversation_observations (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            target_instance_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            snapshot_hash TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            relation TEXT NOT NULL,
            reason TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            keys_json TEXT NOT NULL,
            source_job_id INTEGER,
            capture_generation INTEGER NOT NULL DEFAULT 0,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(target_instance_id, conversation_id, session_id, snapshot_hash),
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(target_instance_id) REFERENCES target_instances(id),
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            FOREIGN KEY(session_id) REFERENCES conversation_sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_observations_anchor
        ON conversation_observations(account_scope_id, conversation_id, session_id, status, observed_at);
        CREATE INDEX idx_observations_expiry ON conversation_observations(expires_at);
        """,
    ),
    (
        18,
        """
        -- A reliable incoming message owns an immutable AI trigger snapshot.
        -- Later page/window changes may affect presentation, but must not erase
        -- or retroactively invalidate the event that was legal at ingestion.
        ALTER TABLE ai_analysis_events ADD COLUMN target_instance_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE ai_analysis_events ADD COLUMN conversation_id TEXT;
        ALTER TABLE ai_analysis_events ADD COLUMN trigger_snapshot_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE ai_analysis_events ADD COLUMN started_at TEXT;
        ALTER TABLE ai_analysis_events ADD COLUMN completed_at TEXT;
        ALTER TABLE ai_analysis_events ADD COLUMN failure_reason TEXT NOT NULL DEFAULT '';
        ALTER TABLE ai_analysis_events ADD COLUMN result_json TEXT NOT NULL DEFAULT '{}';
        CREATE INDEX idx_ai_analysis_events_conversation
        ON ai_analysis_events(account_scope_id, conversation_id, created_at);
        CREATE INDEX idx_ai_analysis_events_status
        ON ai_analysis_events(status, created_at);
        """,
    ),
    (
        19,
        """
        -- Phase 2, step 1: local, structured customer records.  These tables
        -- extend (rather than reinterpret) the Phase 1 capture/timeline tables.
        CREATE TABLE customer_profiles (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            target_instance_id TEXT,
            contact_id TEXT NOT NULL,
            conversation_id TEXT,
            display_name TEXT NOT NULL,
            sales_stage TEXT NOT NULL DEFAULT 'new'
                CHECK(sales_stage IN (
                    'new', 'needs_identified', 'solution_provided',
                    'product_recommended', 'quoted', 'considering', 'ordered',
                    'using', 'result_followup', 'repeat_customer', 'dormant'
                )),
            latest_contact_at TEXT,
            next_followup_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(account_scope_id, contact_id),
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(target_instance_id) REFERENCES target_instances(id) ON DELETE SET NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_profiles_stage
        ON customer_profiles(account_scope_id, sales_stage, updated_at);
        CREATE INDEX idx_customer_profiles_followup
        ON customer_profiles(next_followup_at) WHERE next_followup_at IS NOT NULL;

        CREATE TABLE customer_growing_contexts (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            crop TEXT NOT NULL DEFAULT '',
            variety TEXT NOT NULL DEFAULT '',
            region TEXT NOT NULL DEFAULT '',
            acreage_value REAL CHECK(acreage_value IS NULL OR acreage_value >= 0),
            acreage_unit TEXT NOT NULL DEFAULT '',
            growth_stage TEXT NOT NULL DEFAULT '',
            cultivation_type TEXT NOT NULL DEFAULT '',
            season_or_year TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'uncertain'
                CHECK(status IN ('active', 'historical', 'uncertain')),
            first_seen_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_customer_growing_contexts_profile_status
        ON customer_growing_contexts(customer_profile_id, status, last_confirmed_at);

        CREATE TABLE customer_facts (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            growing_context_id TEXT,
            fact_category TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            value_text TEXT NOT NULL,
            value_json TEXT NOT NULL DEFAULT '{}',
            normalized_value TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            provenance TEXT NOT NULL
                CHECK(provenance IN ('customer_explicit', 'user_explicit', 'ai_inferred', 'system_imported')),
            status TEXT NOT NULL
                CHECK(status IN ('active', 'uncertain', 'conflicting', 'superseded', 'invalid')),
            first_seen_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            conflict_group_id TEXT,
            superseded_by_fact_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(status <> 'active' OR provenance IN ('customer_explicit', 'system_imported')),
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE,
            FOREIGN KEY(growing_context_id) REFERENCES customer_growing_contexts(id) ON DELETE SET NULL,
            FOREIGN KEY(superseded_by_fact_id) REFERENCES customer_facts(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_facts_profile_status
        ON customer_facts(customer_profile_id, status, fact_category, fact_key);
        CREATE INDEX idx_customer_facts_context_status
        ON customer_facts(growing_context_id, status) WHERE growing_context_id IS NOT NULL;
        CREATE INDEX idx_customer_facts_conflict
        ON customer_facts(conflict_group_id) WHERE conflict_group_id IS NOT NULL;
        CREATE INDEX idx_customer_facts_normalized
        ON customer_facts(customer_profile_id, fact_category, fact_key, normalized_value);

        CREATE TABLE customer_fact_sources (
            id TEXT PRIMARY KEY,
            fact_id TEXT NOT NULL,
            message_id TEXT,
            conversation_id TEXT,
            source_text TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(fact_id) REFERENCES customer_facts(id) ON DELETE CASCADE,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_fact_sources_fact
        ON customer_fact_sources(fact_id, observed_at);
        CREATE UNIQUE INDEX idx_customer_fact_sources_message
        ON customer_fact_sources(fact_id, message_id) WHERE message_id IS NOT NULL;
        CREATE UNIQUE INDEX idx_customer_fact_sources_manual
        ON customer_fact_sources(fact_id, IFNULL(conversation_id, ''), source_text, observed_at)
        WHERE message_id IS NULL;

        CREATE TABLE customer_events (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            growing_context_id TEXT,
            event_type TEXT NOT NULL
                CHECK(event_type IN (
                    'problem_reported', 'diagnosis', 'solution_given',
                    'product_recommended', 'dosage_given', 'quote_given',
                    'order_intent', 'order_placed', 'purchased', 'usage_started',
                    'effect_feedback', 'complaint', 'followup',
                    'customer_concern', 'management_plan', 'other'
                )),
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            structured_data_json TEXT NOT NULL DEFAULT '{}',
            event_at TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            provenance TEXT NOT NULL
                CHECK(provenance IN ('customer_explicit', 'user_explicit', 'ai_inferred', 'system_imported')),
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'uncertain', 'superseded', 'invalid')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE,
            FOREIGN KEY(growing_context_id) REFERENCES customer_growing_contexts(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_events_timeline
        ON customer_events(customer_profile_id, event_at, event_type);
        CREATE INDEX idx_customer_events_context
        ON customer_events(growing_context_id, event_at) WHERE growing_context_id IS NOT NULL;

        CREATE TABLE customer_event_sources (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            message_id TEXT,
            conversation_id TEXT,
            source_text TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES customer_events(id) ON DELETE CASCADE,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_event_sources_event
        ON customer_event_sources(event_id, observed_at);
        CREATE UNIQUE INDEX idx_customer_event_sources_message
        ON customer_event_sources(event_id, message_id) WHERE message_id IS NOT NULL;
        CREATE UNIQUE INDEX idx_customer_event_sources_manual
        ON customer_event_sources(event_id, IFNULL(conversation_id, ''), source_text, observed_at)
        WHERE message_id IS NULL;

        CREATE TABLE customer_followups (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            growing_context_id TEXT,
            related_event_id TEXT,
            followup_type TEXT NOT NULL,
            title TEXT NOT NULL,
            due_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'completed', 'cancelled')),
            priority INTEGER NOT NULL DEFAULT 50 CHECK(priority >= 0 AND priority <= 100),
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE,
            FOREIGN KEY(growing_context_id) REFERENCES customer_growing_contexts(id) ON DELETE SET NULL,
            FOREIGN KEY(related_event_id) REFERENCES customer_events(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_followups_pending
        ON customer_followups(status, due_at, priority);
        CREATE INDEX idx_customer_followups_profile
        ON customer_followups(customer_profile_id, status, due_at);
        """,
    ),
    (
        20,
        """
        -- Phase 2, step 2: durable, once-per-message structured memory jobs.
        -- No existing message is backfilled.  The activation cursor is created
        -- by the new worker on its first start at the then-current MAX(rowid).
        CREATE TABLE memory_extraction_jobs (
            id TEXT PRIMARY KEY,
            source_message_id TEXT NOT NULL UNIQUE,
            source_message_rowid INTEGER NOT NULL,
            account_scope_id TEXT NOT NULL,
            target_instance_id TEXT,
            contact_id TEXT NOT NULL,
            conversation_id TEXT,
            customer_profile_id TEXT,
            status TEXT NOT NULL
                CHECK(status IN ('pending', 'running', 'completed', 'no_memory', 'blocked', 'failed', 'invalid_output')),
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            input_hash TEXT NOT NULL DEFAULT '',
            input_context_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT NOT NULL DEFAULT '{}',
            write_result_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            ai_call_count INTEGER NOT NULL DEFAULT 0 CHECK(ai_call_count >= 0 AND ai_call_count <= 1),
            provider_elapsed_ms INTEGER,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(target_instance_id) REFERENCES target_instances(id) ON DELETE SET NULL,
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_memory_extraction_jobs_status
        ON memory_extraction_jobs(status, created_at);
        CREATE INDEX idx_memory_extraction_jobs_identity
        ON memory_extraction_jobs(account_scope_id, contact_id, conversation_id, created_at);

        CREATE TABLE memory_extraction_state (
            id TEXT PRIMARY KEY CHECK(id = 'global'),
            activated_at TEXT NOT NULL,
            last_message_rowid INTEGER NOT NULL DEFAULT 0 CHECK(last_message_rowid >= 0),
            last_scan_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        21,
        """
        -- Phase 2, step 4: manual customer-memory corrections must be
        -- authoritative without destroying the original fact or its sources.
        -- Rebuild the two fact tables so active user_explicit facts are legal
        -- while preserving every v19/v20 id and relationship.
        CREATE TABLE customer_facts_v21 (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            growing_context_id TEXT,
            fact_category TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            value_text TEXT NOT NULL,
            value_json TEXT NOT NULL DEFAULT '{}',
            normalized_value TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            provenance TEXT NOT NULL
                CHECK(provenance IN ('customer_explicit', 'user_explicit', 'ai_inferred', 'system_imported')),
            status TEXT NOT NULL
                CHECK(status IN ('active', 'uncertain', 'conflicting', 'superseded', 'invalid')),
            first_seen_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            valid_from TEXT,
            valid_to TEXT,
            conflict_group_id TEXT,
            superseded_by_fact_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(status <> 'active' OR provenance IN ('customer_explicit', 'user_explicit', 'system_imported')),
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE,
            FOREIGN KEY(growing_context_id) REFERENCES customer_growing_contexts(id) ON DELETE SET NULL,
            FOREIGN KEY(superseded_by_fact_id) REFERENCES customer_facts_v21(id) ON DELETE SET NULL
        );
        INSERT INTO customer_facts_v21
        SELECT * FROM customer_facts;

        CREATE TABLE customer_fact_sources_v21 (
            id TEXT PRIMARY KEY,
            fact_id TEXT NOT NULL,
            message_id TEXT,
            conversation_id TEXT,
            source_text TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(fact_id) REFERENCES customer_facts_v21(id) ON DELETE CASCADE,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );
        INSERT INTO customer_fact_sources_v21
        SELECT * FROM customer_fact_sources;

        DROP TABLE customer_fact_sources;
        DROP TABLE customer_facts;
        ALTER TABLE customer_facts_v21 RENAME TO customer_facts;
        ALTER TABLE customer_fact_sources_v21 RENAME TO customer_fact_sources;

        CREATE INDEX idx_customer_facts_profile_status
        ON customer_facts(customer_profile_id, status, fact_category, fact_key);
        CREATE INDEX idx_customer_facts_context_status
        ON customer_facts(growing_context_id, status) WHERE growing_context_id IS NOT NULL;
        CREATE INDEX idx_customer_facts_conflict
        ON customer_facts(conflict_group_id) WHERE conflict_group_id IS NOT NULL;
        CREATE INDEX idx_customer_facts_normalized
        ON customer_facts(customer_profile_id, fact_category, fact_key, normalized_value);
        CREATE INDEX idx_customer_fact_sources_fact
        ON customer_fact_sources(fact_id, observed_at);
        CREATE UNIQUE INDEX idx_customer_fact_sources_message
        ON customer_fact_sources(fact_id, message_id) WHERE message_id IS NOT NULL;
        CREATE UNIQUE INDEX idx_customer_fact_sources_manual
        ON customer_fact_sources(fact_id, IFNULL(conversation_id, ''), source_text, observed_at)
        WHERE message_id IS NULL;

        CREATE TABLE customer_memory_edits (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            entity_type TEXT NOT NULL CHECK(entity_type IN ('fact', 'growing_context', 'event', 'followup')),
            entity_id TEXT NOT NULL,
            operation TEXT NOT NULL CHECK(operation IN ('confirm', 'correct', 'invalidate')),
            old_value_json TEXT NOT NULL DEFAULT '{}',
            new_value_json TEXT NOT NULL DEFAULT '{}',
            reason TEXT NOT NULL DEFAULT '',
            actor TEXT NOT NULL DEFAULT 'local_user',
            created_at TEXT NOT NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_customer_memory_edits_profile_time
        ON customer_memory_edits(customer_profile_id, created_at DESC);
        CREATE INDEX idx_customer_memory_edits_entity
        ON customer_memory_edits(entity_type, entity_id, created_at DESC);
        """,
    ),
    (
        22,
        """
        -- Phase 2, step 5: local sales-stage provenance/audit and durable,
        -- explainable follow-up sources.  Existing v19-v21 rows are preserved.
        ALTER TABLE customer_profiles ADD COLUMN sales_stage_source TEXT NOT NULL
            DEFAULT 'system_derived'
            CHECK(sales_stage_source IN ('system_derived', 'user_explicit'));
        ALTER TABLE customer_profiles ADD COLUMN sales_stage_updated_at TEXT;

        CREATE TABLE customer_sales_stage_history (
            id TEXT PRIMARY KEY,
            customer_profile_id TEXT NOT NULL,
            old_stage TEXT NOT NULL
                CHECK(old_stage IN (
                    'new', 'needs_identified', 'solution_provided',
                    'product_recommended', 'quoted', 'considering', 'ordered',
                    'using', 'result_followup', 'repeat_customer', 'dormant'
                )),
            new_stage TEXT NOT NULL
                CHECK(new_stage IN (
                    'new', 'needs_identified', 'solution_provided',
                    'product_recommended', 'quoted', 'considering', 'ordered',
                    'using', 'result_followup', 'repeat_customer', 'dormant'
                )),
            reason TEXT NOT NULL DEFAULT '',
            source_event_id TEXT,
            source TEXT NOT NULL
                CHECK(source IN ('system_derived', 'user_explicit')),
            changed_at TEXT NOT NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE CASCADE,
            FOREIGN KEY(source_event_id) REFERENCES customer_events(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_customer_sales_stage_history_profile
        ON customer_sales_stage_history(customer_profile_id, changed_at DESC, id DESC);

        ALTER TABLE customer_followups ADD COLUMN source TEXT NOT NULL
            DEFAULT 'explicit'
            CHECK(source IN ('explicit', 'system_rule', 'user_created'));
        ALTER TABLE customer_followups ADD COLUMN source_message_id TEXT
            REFERENCES messages(id) ON DELETE SET NULL;
        ALTER TABLE customer_followups ADD COLUMN reason TEXT NOT NULL DEFAULT '';
        ALTER TABLE customer_followups ADD COLUMN cancelled_at TEXT;
        CREATE INDEX idx_customer_followups_source
        ON customer_followups(source, related_event_id, source_message_id);
        """,
    ),
    (
        23,
        """
        -- Phase 3, step 1: local, source-traceable product and agronomy
        -- solution knowledge. Catalog knowledge is independent from customer memory.
        CREATE TABLE knowledge_sources (
            source_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK(source_type IN (
                'manual', 'company_document', 'feishu', 'lexiang',
                'historical_solution', 'verified_chat', 'other'
            )),
            title TEXT NOT NULL,
            source_reference TEXT NOT NULL DEFAULT '',
            source_hash TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            imported_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_knowledge_sources_type_status
        ON knowledge_sources(source_type, status, updated_at DESC);
        CREATE UNIQUE INDEX idx_knowledge_sources_reference
        ON knowledge_sources(source_type, source_reference)
        WHERE source_reference <> '';

        CREATE TABLE product_catalog (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            category TEXT NOT NULL DEFAULT '',
            formulation TEXT NOT NULL DEFAULT '',
            active_ingredients_json TEXT NOT NULL DEFAULT '[]',
            ingredient_content TEXT NOT NULL DEFAULT '',
            specifications_json TEXT NOT NULL DEFAULT '{}',
            key_functions_json TEXT NOT NULL DEFAULT '[]',
            applicable_crops_json TEXT NOT NULL DEFAULT '[]',
            applicable_stages_json TEXT NOT NULL DEFAULT '[]',
            applicable_problems_json TEXT NOT NULL DEFAULT '[]',
            usage_methods_json TEXT NOT NULL DEFAULT '[]',
            dosage_text TEXT NOT NULL DEFAULT '',
            dosage_structured_json TEXT NOT NULL DEFAULT '{}',
            precautions_json TEXT NOT NULL DEFAULT '[]',
            incompatibilities_json TEXT NOT NULL DEFAULT '[]',
            risk_notes_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            source_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id)
        );
        CREATE UNIQUE INDEX idx_product_catalog_name_source
        ON product_catalog(normalized_name, source_id);
        CREATE INDEX idx_product_catalog_status_name
        ON product_catalog(status, normalized_name);

        CREATE TABLE product_aliases (
            alias_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            alias TEXT NOT NULL,
            normalized_alias TEXT NOT NULL UNIQUE,
            source_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_catalog(product_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id)
        );
        CREATE INDEX idx_product_aliases_product ON product_aliases(product_id);

        CREATE TABLE product_claims (
            claim_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            claim_type TEXT NOT NULL CHECK(claim_type IN (
                'function', 'crop', 'problem', 'stage', 'dosage',
                'usage', 'precaution', 'compatibility', 'risk',
                'ingredient', 'specification', 'other'
            )),
            claim_value TEXT NOT NULL,
            normalized_value TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_text TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_catalog(product_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(product_id, claim_type, normalized_value, source_id)
        );
        CREATE INDEX idx_product_claims_lookup
        ON product_claims(claim_type, normalized_value, status, product_id);
        CREATE INDEX idx_product_claims_product ON product_claims(product_id, status);

        CREATE TABLE product_versions (
            product_version_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK(version >= 1),
            snapshot_json TEXT NOT NULL,
            source_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_catalog(product_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(product_id, version)
        );
        CREATE INDEX idx_product_versions_product ON product_versions(product_id, version DESC);

        CREATE TABLE solution_templates (
            solution_id TEXT PRIMARY KEY,
            solution_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            crops_json TEXT NOT NULL DEFAULT '[]',
            target_problems_json TEXT NOT NULL DEFAULT '[]',
            growth_stages_json TEXT NOT NULL DEFAULT '[]',
            region_constraints_json TEXT NOT NULL DEFAULT '[]',
            conditions_json TEXT NOT NULL DEFAULT '{}',
            expected_goal TEXT NOT NULL DEFAULT '',
            precautions_json TEXT NOT NULL DEFAULT '[]',
            risk_notes_json TEXT NOT NULL DEFAULT '[]',
            notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            source_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id)
        );
        CREATE UNIQUE INDEX idx_solution_templates_name_source
        ON solution_templates(normalized_name, source_id);
        CREATE INDEX idx_solution_templates_status_name
        ON solution_templates(status, normalized_name);

        CREATE TABLE solution_items (
            solution_item_id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT '',
            dosage TEXT NOT NULL DEFAULT '',
            sequence INTEGER NOT NULL DEFAULT 1 CHECK(sequence >= 1),
            timing TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(solution_id) REFERENCES solution_templates(solution_id) ON DELETE CASCADE,
            FOREIGN KEY(product_id) REFERENCES product_catalog(product_id),
            UNIQUE(solution_id, sequence)
        );
        CREATE INDEX idx_solution_items_solution ON solution_items(solution_id, sequence);
        CREATE INDEX idx_solution_items_product ON solution_items(product_id, solution_id);

        CREATE TABLE solution_versions (
            solution_version_id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            version INTEGER NOT NULL CHECK(version >= 1),
            snapshot_json TEXT NOT NULL,
            source_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(solution_id) REFERENCES solution_templates(solution_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(solution_id, version)
        );
        CREATE INDEX idx_solution_versions_solution ON solution_versions(solution_id, version DESC);
        """,
    ),
    (
        24,
        """
        -- Phase 3, step 2: source-by-source evidence, conservative review
        -- gates, unresolved solution items, and deterministic import audit.
        ALTER TABLE product_catalog ADD COLUMN review_state TEXT NOT NULL
            DEFAULT 'verified'
            CHECK(review_state IN ('verified', 'unreviewed', 'review_required', 'conflicting'));
        ALTER TABLE product_claims ADD COLUMN review_state TEXT NOT NULL
            DEFAULT 'verified'
            CHECK(review_state IN ('verified', 'unreviewed', 'review_required', 'conflicting'));
        ALTER TABLE product_claims ADD COLUMN conflict_group_id TEXT;
        ALTER TABLE product_claims ADD COLUMN dosage_structured_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE solution_templates ADD COLUMN review_state TEXT NOT NULL
            DEFAULT 'verified'
            CHECK(review_state IN ('verified', 'unreviewed', 'review_required', 'conflicting'));

        CREATE TABLE product_source_links (
            product_source_link_id TEXT PRIMARY KEY,
            product_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_product_name TEXT NOT NULL DEFAULT '',
            source_version TEXT NOT NULL DEFAULT '',
            source_text TEXT NOT NULL DEFAULT '',
            source_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            review_state TEXT NOT NULL DEFAULT 'unreviewed'
                CHECK(review_state IN ('verified', 'unreviewed', 'review_required', 'conflicting')),
            imported_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES product_catalog(product_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(product_id, source_id)
        );
        CREATE INDEX idx_product_source_links_product
        ON product_source_links(product_id, review_state, source_id);
        CREATE INDEX idx_product_source_links_source
        ON product_source_links(source_id, product_id);

        CREATE TABLE product_alias_sources (
            product_alias_source_id TEXT PRIMARY KEY,
            alias_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(alias_id) REFERENCES product_aliases(alias_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(alias_id, source_id)
        );
        CREATE INDEX idx_product_alias_sources_alias
        ON product_alias_sources(alias_id, source_id);

        CREATE TABLE solution_source_links (
            solution_source_link_id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_solution_name TEXT NOT NULL DEFAULT '',
            source_version TEXT NOT NULL DEFAULT '',
            source_text TEXT NOT NULL DEFAULT '',
            source_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'inactive', 'draft', 'deprecated')),
            review_state TEXT NOT NULL DEFAULT 'unreviewed'
                CHECK(review_state IN ('verified', 'unreviewed', 'review_required', 'conflicting')),
            imported_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(solution_id) REFERENCES solution_templates(solution_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(solution_id, source_id)
        );
        CREATE INDEX idx_solution_source_links_solution
        ON solution_source_links(solution_id, review_state, source_id);

        CREATE TABLE solution_unresolved_items (
            unresolved_item_id TEXT PRIMARY KEY,
            solution_id TEXT NOT NULL,
            raw_product_name TEXT NOT NULL,
            normalized_product_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT '',
            dosage TEXT NOT NULL DEFAULT '',
            sequence INTEGER NOT NULL CHECK(sequence >= 1),
            timing TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            source_id TEXT NOT NULL,
            review_state TEXT NOT NULL DEFAULT 'review_required'
                CHECK(review_state IN ('unreviewed', 'review_required', 'conflicting')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(solution_id) REFERENCES solution_templates(solution_id) ON DELETE CASCADE,
            FOREIGN KEY(source_id) REFERENCES knowledge_sources(source_id),
            UNIQUE(solution_id, sequence)
        );
        CREATE INDEX idx_solution_unresolved_items_solution
        ON solution_unresolved_items(solution_id, sequence);

        CREATE TABLE knowledge_import_runs (
            import_run_id TEXT PRIMARY KEY,
            source_format TEXT NOT NULL,
            preview_hash TEXT NOT NULL UNIQUE,
            preview_json TEXT NOT NULL,
            result_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN ('previewed', 'committed', 'failed')),
            created_at TEXT NOT NULL,
            committed_at TEXT
        );
        CREATE INDEX idx_knowledge_import_runs_status
        ON knowledge_import_runs(status, created_at DESC);

        -- Existing v23 rows remain eligible exactly as before.  New v24 imports
        -- explicitly set their review state and source links.
        CREATE INDEX idx_product_catalog_review
        ON product_catalog(status, review_state, normalized_name);
        CREATE INDEX idx_product_claims_review
        ON product_claims(product_id, status, review_state, claim_type);
        CREATE INDEX idx_solution_templates_review
        ON solution_templates(status, review_state, normalized_name);
        """,
    ),
    (
        25,
        """
        -- Phase 3, step 3: distinguish company SKUs from generic agronomy
        -- categories/ingredients. Existing v24 rows remain company products
        -- and therefore cannot silently become Reply AI recommendation facts.
        ALTER TABLE product_catalog ADD COLUMN knowledge_kind TEXT NOT NULL
            DEFAULT 'company_product'
            CHECK(knowledge_kind IN (
                'company_product', 'generic_category',
                'active_ingredient', 'functional_ingredient'
            ));
        CREATE INDEX idx_product_catalog_kind_review
        ON product_catalog(knowledge_kind, status, review_state, normalized_name);
        """,
    ),
    (
        26,
        """
        -- Phase 4, step 1: durable agronomy coverage/gap observations and
        -- reply-pipeline timing. Customer chat discovers missing knowledge;
        -- it never creates active agronomy knowledge.
        CREATE TABLE knowledge_coverage_evaluations (
            id TEXT PRIMARY KEY,
            source_message_id TEXT NOT NULL UNIQUE,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            crop TEXT NOT NULL DEFAULT '',
            growth_stage TEXT NOT NULL DEFAULT '',
            problem TEXT NOT NULL DEFAULT '',
            issue_type TEXT NOT NULL DEFAULT 'unknown',
            issue_text TEXT NOT NULL DEFAULT '',
            matched_product_ids_json TEXT NOT NULL DEFAULT '[]',
            matched_solution_ids_json TEXT NOT NULL DEFAULT '[]',
            missing_types_json TEXT NOT NULL DEFAULT '[]',
            reply_capability TEXT NOT NULL
                CHECK(reply_capability IN ('direct_direction','partial_direction','clarification_only')),
            affects_reply INTEGER NOT NULL DEFAULT 0 CHECK(affects_reply IN (0,1)),
            evaluated_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_knowledge_coverage_recent
        ON knowledge_coverage_evaluations(evaluated_at DESC, reply_capability);
        CREATE INDEX idx_knowledge_coverage_identity
        ON knowledge_coverage_evaluations(account_scope_id, contact_id, conversation_id, evaluated_at DESC);

        CREATE TABLE knowledge_gaps (
            id TEXT PRIMARY KEY,
            signature_hash TEXT NOT NULL UNIQUE,
            crop TEXT NOT NULL DEFAULT '',
            growth_stage TEXT NOT NULL DEFAULT '',
            problem TEXT NOT NULL DEFAULT '',
            issue_type TEXT NOT NULL DEFAULT 'unknown',
            issue_text TEXT NOT NULL DEFAULT '',
            missing_types_json TEXT NOT NULL DEFAULT '[]',
            available_knowledge_json TEXT NOT NULL DEFAULT '{}',
            occurrence_count INTEGER NOT NULL DEFAULT 0 CHECK(occurrence_count >= 0),
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            affects_reply INTEGER NOT NULL DEFAULT 0 CHECK(affects_reply IN (0,1)),
            status TEXT NOT NULL DEFAULT 'open'
                CHECK(status IN ('open','resolved','ignored')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_knowledge_gaps_priority
        ON knowledge_gaps(status, affects_reply DESC, occurrence_count DESC, last_seen_at DESC);

        CREATE TABLE knowledge_gap_occurrences (
            id TEXT PRIMARY KEY,
            gap_id TEXT NOT NULL,
            evaluation_id TEXT NOT NULL,
            source_message_id TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(gap_id) REFERENCES knowledge_gaps(id) ON DELETE CASCADE,
            FOREIGN KEY(evaluation_id) REFERENCES knowledge_coverage_evaluations(id) ON DELETE CASCADE,
            FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            UNIQUE(gap_id, source_message_id)
        );
        CREATE INDEX idx_knowledge_gap_occurrences_gap
        ON knowledge_gap_occurrences(gap_id, observed_at DESC);

        CREATE TRIGGER trg_knowledge_gap_occurrence_delete
        AFTER DELETE ON knowledge_gap_occurrences
        BEGIN
            UPDATE knowledge_gaps
               SET occurrence_count = MAX(0, occurrence_count - 1),
                   updated_at = CURRENT_TIMESTAMP
             WHERE id = OLD.gap_id;
            DELETE FROM knowledge_gaps
             WHERE id = OLD.gap_id AND occurrence_count = 0;
        END;

        CREATE TABLE reply_pipeline_metrics (
            event_id TEXT PRIMARY KEY,
            source_message_id TEXT NOT NULL UNIQUE,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_observed_at TEXT NOT NULL DEFAULT '',
            event_created_at TEXT NOT NULL DEFAULT '',
            context_started_at TEXT,
            context_completed_at TEXT,
            reply_started_at TEXT,
            reply_completed_at TEXT,
            ui_ready_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES ai_analysis_events(id) ON DELETE CASCADE,
            FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_reply_pipeline_metrics_recent
        ON reply_pipeline_metrics(event_created_at DESC, account_scope_id, contact_id);
        """,
    ),
    (
        27,
        """
        -- Phase 4, step 2: local end-to-end performance trace.  All values are
        -- local timestamps/timings; prompts, credentials and chat text are not
        -- duplicated into this table.
        ALTER TABLE capture_samples ADD COLUMN capture_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN layout_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN title_input_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN content_input_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN title_preprocess_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN title_inference_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN title_postprocess_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN content_preprocess_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN content_inference_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN content_postprocess_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN verify_capture_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN page_classification_elapsed_ms INTEGER;
        ALTER TABLE capture_samples ADD COLUMN title_cache_hit INTEGER NOT NULL DEFAULT 0
            CHECK(title_cache_hit IN (0,1));

        ALTER TABLE reply_pipeline_metrics ADD COLUMN capture_cycle_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN capture_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN capture_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN title_ocr_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN title_ocr_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN content_ocr_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN content_ocr_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN identity_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN identity_stable_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN message_classified_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN message_committed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN reply_event_created_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN reply_queued_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN reply_worker_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN provider_request_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN provider_first_response_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN provider_response_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN reply_parse_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN result_persisted_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN ui_event_received_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN ui_render_started_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN ui_render_completed_at TEXT;
        ALTER TABLE reply_pipeline_metrics ADD COLUMN prompt_profile_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE reply_pipeline_metrics ADD COLUMN provider_profile_json TEXT NOT NULL DEFAULT '{}';
        ALTER TABLE reply_pipeline_metrics ADD COLUMN trace_json TEXT NOT NULL DEFAULT '{}';

        ALTER TABLE memory_extraction_jobs ADD COLUMN provider_started_at TEXT;
        ALTER TABLE memory_extraction_jobs ADD COLUMN provider_completed_at TEXT;
        ALTER TABLE memory_extraction_jobs ADD COLUMN write_started_at TEXT;
        ALTER TABLE memory_extraction_jobs ADD COLUMN write_completed_at TEXT;
        """,
    ),
    (
        28,
        """
        -- Phase 4, step 3: one-time, local historical chat archive and
        -- resumable customer-memory backfill.  Historical rows deliberately
        -- live outside messages, so they cannot trigger the live OCR/reply/
        -- memory cursor or alter runtime conversation anchors.
        CREATE TABLE historical_import_sources (
            id TEXT PRIMARY KEY,
            source_kind TEXT NOT NULL CHECK(source_kind IN (
                'fixture_json','export_json','export_csv','visible_transcript','other'
            )),
            display_name TEXT NOT NULL,
            source_reference TEXT NOT NULL DEFAULT '',
            adapter_version TEXT NOT NULL DEFAULT '1',
            source_hash TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX idx_historical_import_sources_reference
        ON historical_import_sources(source_kind, source_reference)
        WHERE source_reference <> '';

        CREATE TABLE historical_contact_mappings (
            id TEXT PRIMARY KEY,
            import_source_id TEXT NOT NULL,
            mapping_key TEXT NOT NULL,
            account_scope_id TEXT NOT NULL,
            external_contact_id TEXT NOT NULL DEFAULT '',
            external_conversation_id TEXT NOT NULL DEFAULT '',
            source_display_name TEXT NOT NULL DEFAULT '',
            contact_id TEXT,
            conversation_id TEXT,
            status TEXT NOT NULL CHECK(status IN (
                'resolved','probable','unresolved','conflicting'
            )),
            match_basis TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence >= 0 AND confidence <= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(import_source_id) REFERENCES historical_import_sources(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            UNIQUE(import_source_id, account_scope_id, mapping_key)
        );
        CREATE INDEX idx_historical_contact_mappings_status
        ON historical_contact_mappings(account_scope_id, status, source_display_name);
        CREATE INDEX idx_historical_contact_mappings_contact
        ON historical_contact_mappings(account_scope_id, contact_id, status);

        CREATE TABLE historical_import_jobs (
            id TEXT PRIMARY KEY,
            import_source_id TEXT NOT NULL,
            mapping_id TEXT,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT,
            conversation_id TEXT,
            start_time TEXT,
            end_time TEXT,
            message_count INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
            duplicate_count INTEGER NOT NULL DEFAULT 0 CHECK(duplicate_count >= 0),
            committed_count INTEGER NOT NULL DEFAULT 0 CHECK(committed_count >= 0),
            status TEXT NOT NULL CHECK(status IN (
                'previewed','pending','running','paused','completed','failed','blocked'
            )),
            checkpoint_sequence INTEGER NOT NULL DEFAULT -1,
            preview_hash TEXT NOT NULL,
            preview_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(import_source_id) REFERENCES historical_import_sources(id) ON DELETE CASCADE,
            FOREIGN KEY(mapping_id) REFERENCES historical_contact_mappings(id) ON DELETE SET NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            UNIQUE(import_source_id, account_scope_id, preview_hash)
        );
        CREATE INDEX idx_historical_import_jobs_status
        ON historical_import_jobs(status, updated_at DESC);
        CREATE INDEX idx_historical_import_jobs_contact
        ON historical_import_jobs(account_scope_id, contact_id, created_at DESC);

        CREATE TABLE historical_chat_messages (
            historical_message_id TEXT PRIMARY KEY,
            import_source_id TEXT NOT NULL,
            import_job_id TEXT NOT NULL,
            mapping_id TEXT NOT NULL,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT,
            conversation_id TEXT,
            external_contact_id TEXT NOT NULL DEFAULT '',
            external_conversation_id TEXT NOT NULL DEFAULT '',
            external_message_id TEXT NOT NULL DEFAULT '',
            sender TEXT NOT NULL CHECK(sender IN ('other','me','system','unknown')),
            message_type TEXT NOT NULL DEFAULT 'text',
            text TEXT NOT NULL DEFAULT '',
            sent_at TEXT NOT NULL,
            source_sequence INTEGER NOT NULL,
            source_hash TEXT NOT NULL,
            mapping_status TEXT NOT NULL CHECK(mapping_status IN (
                'resolved','probable','unresolved','conflicting'
            )),
            imported_at TEXT NOT NULL,
            raw_metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY(import_source_id) REFERENCES historical_import_sources(id) ON DELETE CASCADE,
            FOREIGN KEY(import_job_id) REFERENCES historical_import_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(mapping_id) REFERENCES historical_contact_mappings(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            UNIQUE(import_source_id, account_scope_id, source_hash)
        );
        CREATE INDEX idx_historical_messages_contact_time
        ON historical_chat_messages(account_scope_id, contact_id, sent_at, source_sequence);
        CREATE INDEX idx_historical_messages_mapping_time
        ON historical_chat_messages(mapping_id, sent_at, source_sequence);
        CREATE INDEX idx_historical_messages_job_sequence
        ON historical_chat_messages(import_job_id, source_sequence);

        CREATE TABLE historical_memory_backfill_jobs (
            id TEXT PRIMARY KEY,
            import_job_id TEXT NOT NULL,
            mapping_id TEXT NOT NULL,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            conversation_id TEXT,
            customer_profile_id TEXT,
            status TEXT NOT NULL CHECK(status IN (
                'pending','running','paused','completed','partial','failed','blocked'
            )),
            total_messages INTEGER NOT NULL DEFAULT 0 CHECK(total_messages >= 0),
            eligible_messages INTEGER NOT NULL DEFAULT 0 CHECK(eligible_messages >= 0),
            total_chunks INTEGER NOT NULL DEFAULT 0 CHECK(total_chunks >= 0),
            completed_chunks INTEGER NOT NULL DEFAULT 0 CHECK(completed_chunks >= 0),
            checkpoint_chunk INTEGER NOT NULL DEFAULT -1,
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(import_job_id) REFERENCES historical_import_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(mapping_id) REFERENCES historical_contact_mappings(id) ON DELETE CASCADE,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            FOREIGN KEY(customer_profile_id) REFERENCES customer_profiles(id) ON DELETE SET NULL,
            UNIQUE(import_job_id, contact_id)
        );
        CREATE INDEX idx_historical_backfill_jobs_status
        ON historical_memory_backfill_jobs(status, updated_at DESC);
        CREATE INDEX idx_historical_backfill_jobs_contact
        ON historical_memory_backfill_jobs(account_scope_id, contact_id, created_at DESC);

        CREATE TABLE historical_memory_backfill_chunks (
            id TEXT PRIMARY KEY,
            backfill_job_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL CHECK(chunk_index >= 0),
            first_message_id TEXT NOT NULL,
            last_message_id TEXT NOT NULL,
            message_count INTEGER NOT NULL CHECK(message_count > 0),
            character_count INTEGER NOT NULL CHECK(character_count >= 0),
            input_hash TEXT NOT NULL,
            input_context_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT NOT NULL DEFAULT '{}',
            write_result_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN (
                'pending','running','completed','no_memory','invalid_output','failed','blocked'
            )),
            ai_call_count INTEGER NOT NULL DEFAULT 0 CHECK(ai_call_count >= 0 AND ai_call_count <= 1),
            error TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(backfill_job_id) REFERENCES historical_memory_backfill_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(first_message_id) REFERENCES historical_chat_messages(historical_message_id),
            FOREIGN KEY(last_message_id) REFERENCES historical_chat_messages(historical_message_id),
            UNIQUE(backfill_job_id, chunk_index),
            UNIQUE(backfill_job_id, input_hash)
        );
        CREATE INDEX idx_historical_backfill_chunks_status
        ON historical_memory_backfill_chunks(backfill_job_id, status, chunk_index);

        ALTER TABLE customer_profiles ADD COLUMN history_status TEXT NOT NULL
            DEFAULT 'history_not_initialized'
            CHECK(history_status IN (
                'history_not_initialized','history_imported','memory_backfill_pending',
                'memory_backfilled','partial','failed'
            ));
        ALTER TABLE customer_profiles ADD COLUMN history_last_imported_at TEXT;
        ALTER TABLE customer_profiles ADD COLUMN history_last_backfilled_at TEXT;
        ALTER TABLE customer_profiles ADD COLUMN history_status_detail TEXT NOT NULL DEFAULT '';

        ALTER TABLE customer_fact_sources ADD COLUMN historical_message_id TEXT
            REFERENCES historical_chat_messages(historical_message_id) ON DELETE SET NULL;
        ALTER TABLE customer_event_sources ADD COLUMN historical_message_id TEXT
            REFERENCES historical_chat_messages(historical_message_id) ON DELETE SET NULL;
        ALTER TABLE customer_followups ADD COLUMN source_historical_message_id TEXT
            REFERENCES historical_chat_messages(historical_message_id) ON DELETE SET NULL;

        CREATE UNIQUE INDEX idx_customer_fact_sources_historical
        ON customer_fact_sources(fact_id, historical_message_id)
        WHERE historical_message_id IS NOT NULL;
        CREATE UNIQUE INDEX idx_customer_event_sources_historical
        ON customer_event_sources(event_id, historical_message_id)
        WHERE historical_message_id IS NOT NULL;
        CREATE INDEX idx_customer_followups_historical_source
        ON customer_followups(source_historical_message_id)
        WHERE source_historical_message_id IS NOT NULL;
        """,
    ),
    (
        29,
        """
        -- Phase 4, step 3 production history migration metadata.  These
        -- records describe a one-time, local source snapshot.  Daily WhoChat
        -- runtime never reads the decrypted source databases.
        CREATE TABLE historical_source_accounts (
            id TEXT PRIMARY KEY,
            source_account_id TEXT NOT NULL UNIQUE,
            stable_identifier TEXT NOT NULL UNIQUE,
            public_alias TEXT NOT NULL DEFAULT '',
            source_data_directory TEXT NOT NULL,
            account_scope_id TEXT,
            target_instance_id TEXT,
            mapping_status TEXT NOT NULL CHECK(mapping_status IN (
                'resolved','unresolved','conflicting'
            )),
            match_basis TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0 CHECK(confidence >= 0 AND confidence <= 1),
            manifest_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id) ON DELETE SET NULL,
            FOREIGN KEY(target_instance_id) REFERENCES target_instances(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_historical_source_accounts_scope
        ON historical_source_accounts(account_scope_id, mapping_status);

        CREATE TABLE historical_snapshots (
            id TEXT PRIMARY KEY,
            source_account_record_id TEXT NOT NULL,
            import_source_id TEXT,
            snapshot_label TEXT NOT NULL,
            source_directory TEXT NOT NULL,
            decrypted_directory TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'discovered','decrypted','verified','normalized','imported','failed'
            )),
            snapshot_started_at TEXT NOT NULL,
            snapshot_completed_at TEXT,
            latest_historical_message_at TEXT,
            cutover_at TEXT,
            database_count INTEGER NOT NULL DEFAULT 0 CHECK(database_count >= 0),
            database_size_bytes INTEGER NOT NULL DEFAULT 0 CHECK(database_size_bytes >= 0),
            message_count INTEGER NOT NULL DEFAULT 0 CHECK(message_count >= 0),
            snapshot_hash TEXT NOT NULL DEFAULT '',
            manifest_json TEXT NOT NULL DEFAULT '{}',
            key_material_status TEXT NOT NULL DEFAULT 'pending_cleanup' CHECK(key_material_status IN (
                'pending_cleanup','deleted','retained_for_recovery','not_created'
            )),
            key_material_deleted_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_account_record_id) REFERENCES historical_source_accounts(id) ON DELETE CASCADE,
            FOREIGN KEY(import_source_id) REFERENCES historical_import_sources(id) ON DELETE SET NULL,
            UNIQUE(source_account_record_id, snapshot_label)
        );
        CREATE INDEX idx_historical_snapshots_status
        ON historical_snapshots(status, cutover_at);

        CREATE TABLE historical_gap_candidates (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT,
            conversation_id TEXT,
            gap_start_at TEXT,
            gap_end_at TEXT,
            reason TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN ('detected','reviewed','ignored','supplemented')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_historical_gap_candidates_scope_status
        ON historical_gap_candidates(account_scope_id, status, gap_start_at);

        ALTER TABLE historical_chat_messages ADD COLUMN snapshot_id TEXT
            REFERENCES historical_snapshots(id) ON DELETE SET NULL;
        CREATE INDEX idx_historical_messages_snapshot_sequence
        ON historical_chat_messages(snapshot_id, source_sequence);
        CREATE INDEX idx_historical_messages_external_lookup
        ON historical_chat_messages(account_scope_id, external_conversation_id, sent_at, source_sequence);
        """,
    ),
    (
        30,
        """
        -- One explicitly authorised retry is allowed for a failed historical
        -- AI chunk.  Business logic still requires a failed/invalid first
        -- attempt and blocks any third call; this CHECK only makes the audited
        -- cumulative counter capable of representing the authorised retry.
        CREATE TABLE historical_memory_backfill_chunks_v30 (
            id TEXT PRIMARY KEY,
            backfill_job_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL CHECK(chunk_index >= 0),
            first_message_id TEXT NOT NULL,
            last_message_id TEXT NOT NULL,
            message_count INTEGER NOT NULL CHECK(message_count > 0),
            character_count INTEGER NOT NULL CHECK(character_count >= 0),
            input_hash TEXT NOT NULL,
            input_context_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT NOT NULL DEFAULT '{}',
            write_result_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN (
                'pending','running','completed','no_memory','invalid_output','failed','blocked'
            )),
            ai_call_count INTEGER NOT NULL DEFAULT 0 CHECK(ai_call_count >= 0 AND ai_call_count <= 2),
            error TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(backfill_job_id) REFERENCES historical_memory_backfill_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(first_message_id) REFERENCES historical_chat_messages(historical_message_id),
            FOREIGN KEY(last_message_id) REFERENCES historical_chat_messages(historical_message_id),
            UNIQUE(backfill_job_id, chunk_index),
            UNIQUE(backfill_job_id, input_hash)
        );
        INSERT INTO historical_memory_backfill_chunks_v30(
            id,backfill_job_id,chunk_index,first_message_id,last_message_id,
            message_count,character_count,input_hash,input_context_json,
            output_json,write_result_json,status,ai_call_count,error,
            started_at,completed_at,created_at,updated_at
        )
        SELECT
            id,backfill_job_id,chunk_index,first_message_id,last_message_id,
            message_count,character_count,input_hash,input_context_json,
            output_json,write_result_json,status,ai_call_count,error,
            started_at,completed_at,created_at,updated_at
        FROM historical_memory_backfill_chunks;
        DROP TABLE historical_memory_backfill_chunks;
        ALTER TABLE historical_memory_backfill_chunks_v30
            RENAME TO historical_memory_backfill_chunks;
        CREATE INDEX idx_historical_backfill_chunks_status
        ON historical_memory_backfill_chunks(backfill_job_id, status, chunk_index);
        """,
    ),
    (
        31,
        """
        -- Historical provider timeouts are resumable and distinct from
        -- validation failures. Each chunk remains capped at two provider calls
        -- (initial + one retry), while later chunks can continue independently.
        CREATE TABLE historical_memory_backfill_chunks_v31 (
            id TEXT PRIMARY KEY,
            backfill_job_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL CHECK(chunk_index >= 0),
            first_message_id TEXT NOT NULL,
            last_message_id TEXT NOT NULL,
            message_count INTEGER NOT NULL CHECK(message_count > 0),
            character_count INTEGER NOT NULL CHECK(character_count >= 0),
            input_hash TEXT NOT NULL,
            input_context_json TEXT NOT NULL DEFAULT '{}',
            output_json TEXT NOT NULL DEFAULT '{}',
            write_result_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL CHECK(status IN (
                'pending','running','completed','no_memory','invalid_output','failed',
                'blocked','provider_timeout_pending'
            )),
            ai_call_count INTEGER NOT NULL DEFAULT 0 CHECK(ai_call_count >= 0 AND ai_call_count <= 2),
            error TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            completed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(backfill_job_id) REFERENCES historical_memory_backfill_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(first_message_id) REFERENCES historical_chat_messages(historical_message_id),
            FOREIGN KEY(last_message_id) REFERENCES historical_chat_messages(historical_message_id),
            UNIQUE(backfill_job_id, chunk_index),
            UNIQUE(backfill_job_id, input_hash)
        );
        INSERT INTO historical_memory_backfill_chunks_v31(
            id,backfill_job_id,chunk_index,first_message_id,last_message_id,
            message_count,character_count,input_hash,input_context_json,
            output_json,write_result_json,status,ai_call_count,error,
            started_at,completed_at,created_at,updated_at
        )
        SELECT
            id,backfill_job_id,chunk_index,first_message_id,last_message_id,
            message_count,character_count,input_hash,input_context_json,
            output_json,write_result_json,
            CASE
                WHEN status='failed'
                 AND error LIKE '%historical_provider_transport_error:TimeoutError%'
                THEN 'provider_timeout_pending'
                ELSE status
            END,
            ai_call_count,error,started_at,completed_at,created_at,updated_at
        FROM historical_memory_backfill_chunks;
        DROP TABLE historical_memory_backfill_chunks;
        ALTER TABLE historical_memory_backfill_chunks_v31
            RENAME TO historical_memory_backfill_chunks;
        CREATE INDEX idx_historical_backfill_chunks_status
        ON historical_memory_backfill_chunks(backfill_job_id, status, chunk_index);
        """,
    ),
    (
        32,
        """
        -- Reply context reads a bounded recent-chat tail.  Keep the customer
        -- and account identity in the access path and let SQLite satisfy the
        -- chronological ORDER BY/LIMIT without loading a full conversation.
        CREATE INDEX idx_messages_scope_contact_conversation_observed
        ON messages(account_scope_id,contact_id,conversation_id,observed_at DESC);
        """,
    ),
    (
        33,
        """
        -- Phase 4, step 10: persistent orchestration state for bounded,
        -- low-priority Historical Backfill.  Customer memory extraction stays
        -- in the existing v28-v31 job/chunk tables; these tables only record
        -- batch boundaries, pause/recovery state, budgets and incidents.
        CREATE TABLE historical_scheduler_control (
            id TEXT PRIMARY KEY CHECK(id='default'),
            enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
            manual_paused INTEGER NOT NULL DEFAULT 1 CHECK(manual_paused IN (0,1)),
            automatic_paused INTEGER NOT NULL DEFAULT 0 CHECK(automatic_paused IN (0,1)),
            pause_reason TEXT NOT NULL DEFAULT '',
            current_batch_id TEXT,
            last_batch_local_date TEXT,
            observation_batches_completed INTEGER NOT NULL DEFAULT 0
                CHECK(observation_batches_completed >= 0),
            updated_at TEXT NOT NULL,
            FOREIGN KEY(current_batch_id) REFERENCES historical_scheduler_batches(id)
                DEFERRABLE INITIALLY DEFERRED
        );

        CREATE TABLE historical_scheduler_batches (
            id TEXT PRIMARY KEY,
            sequence_no INTEGER NOT NULL UNIQUE CHECK(sequence_no > 0),
            phase TEXT NOT NULL DEFAULT 'ab_observation',
            status TEXT NOT NULL CHECK(status IN (
                'selecting','prepared','authorized','running','paused_manual',
                'paused_automatic','paused_realtime','paused_window','paused_budget',
                'paused_quality','paused_technical','completed','failed'
            )),
            observation_number INTEGER NOT NULL DEFAULT 0 CHECK(observation_number >= 0),
            scheduled_local_date TEXT,
            selected_customers INTEGER NOT NULL DEFAULT 0 CHECK(selected_customers >= 0),
            planned_chunks INTEGER NOT NULL DEFAULT 0 CHECK(planned_chunks >= 0),
            provider_call_budget INTEGER NOT NULL DEFAULT 0 CHECK(provider_call_budget >= 0),
            provider_calls_used INTEGER NOT NULL DEFAULT 0 CHECK(provider_calls_used >= 0),
            max_runtime_seconds INTEGER NOT NULL DEFAULT 7200 CHECK(max_runtime_seconds > 0),
            config_snapshot_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            stop_reason TEXT NOT NULL DEFAULT '',
            report_json_path TEXT NOT NULL DEFAULT '',
            report_markdown_path TEXT NOT NULL DEFAULT '',
            database_size_before INTEGER CHECK(database_size_before IS NULL OR database_size_before >= 0),
            database_size_after INTEGER CHECK(database_size_after IS NULL OR database_size_after >= 0),
            created_at TEXT NOT NULL,
            authorized_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_historical_scheduler_batches_status
        ON historical_scheduler_batches(status, sequence_no DESC);
        CREATE INDEX idx_historical_scheduler_batches_date
        ON historical_scheduler_batches(scheduled_local_date, status);

        CREATE TABLE historical_scheduler_batch_customers (
            batch_id TEXT NOT NULL,
            safe_contact_reference TEXT NOT NULL,
            mapping_id TEXT NOT NULL,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT,
            conversation_id TEXT,
            backfill_job_id TEXT,
            value_tier TEXT NOT NULL CHECK(value_tier IN ('A','B')),
            priority_score INTEGER NOT NULL,
            priority_rank INTEGER NOT NULL CHECK(priority_rank > 0),
            expected_chunks INTEGER NOT NULL CHECK(expected_chunks > 0),
            state TEXT NOT NULL CHECK(state IN (
                'selected','identity_validated','queued','running','completed','no_memory',
                'needs_review','failed_retryable','failed_terminal'
            )),
            identity_basis TEXT NOT NULL DEFAULT '',
            identity_evidence_json TEXT NOT NULL DEFAULT '{}',
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(batch_id, safe_contact_reference),
            UNIQUE(mapping_id),
            FOREIGN KEY(batch_id) REFERENCES historical_scheduler_batches(id) ON DELETE CASCADE,
            FOREIGN KEY(mapping_id) REFERENCES historical_contact_mappings(id),
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE SET NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            FOREIGN KEY(backfill_job_id) REFERENCES historical_memory_backfill_jobs(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_historical_scheduler_customers_state
        ON historical_scheduler_batch_customers(batch_id, state, priority_rank);
        CREATE INDEX idx_historical_scheduler_customers_scope
        ON historical_scheduler_batch_customers(account_scope_id, state);

        CREATE TABLE historical_scheduler_provider_calls (
            id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL,
            backfill_job_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            attempt_no INTEGER NOT NULL CHECK(attempt_no IN (1,2)),
            call_kind TEXT NOT NULL CHECK(call_kind IN ('initial','transport_retry')),
            outcome TEXT NOT NULL CHECK(outcome IN (
                'started','completed','no_memory','timeout','transport_error',
                'invalid_output','failed'
            )),
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            completed_at TEXT,
            elapsed_ms INTEGER CHECK(elapsed_ms IS NULL OR elapsed_ms >= 0),
            error_class TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(chunk_id, attempt_no),
            FOREIGN KEY(batch_id) REFERENCES historical_scheduler_batches(id) ON DELETE CASCADE,
            FOREIGN KEY(backfill_job_id) REFERENCES historical_memory_backfill_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY(chunk_id) REFERENCES historical_memory_backfill_chunks(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_historical_scheduler_calls_day
        ON historical_scheduler_provider_calls(started_at, outcome);
        CREATE INDEX idx_historical_scheduler_calls_batch
        ON historical_scheduler_provider_calls(batch_id, started_at);

        CREATE TABLE historical_scheduler_incidents (
            id TEXT PRIMARY KEY,
            batch_id TEXT,
            incident_type TEXT NOT NULL CHECK(incident_type IN ('technical','quality')),
            severity TEXT NOT NULL CHECK(severity IN ('warning','circuit_breaker')),
            reason TEXT NOT NULL,
            evidence_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            resolved_at TEXT,
            resolution_note TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(batch_id) REFERENCES historical_scheduler_batches(id) ON DELETE SET NULL
        );
        CREATE INDEX idx_historical_scheduler_incidents_open
        ON historical_scheduler_incidents(resolved_at, created_at DESC);

        INSERT INTO historical_scheduler_control(
            id,enabled,manual_paused,automatic_paused,pause_reason,
            observation_batches_completed,updated_at
        ) VALUES ('default',0,1,0,'awaiting_first_long_running_batch_authorization',0,CURRENT_TIMESTAMP);
        """,
    ),
    (
        34,
        """
        -- Proactive Contact V1 keeps one same-day result for one exact,
        -- account-scoped context hash.  The row stores only the local decision
        -- and generated suggestions; raw prompt payloads and chat text are not
        -- copied into audit logs.
        CREATE TABLE proactive_contact_cache (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            local_date TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            reason_type TEXT NOT NULL CHECK(reason_type IN (
                'reply_obligation','order_followup','usage_followup','effect_followup',
                'predicted_window','reactivation','do_not_contact_now'
            )),
            decision_status TEXT NOT NULL CHECK(decision_status IN (
                'ineligible','do_not_contact','reply_priority','proactive','blocked'
            )),
            outcome_json TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            suggestion_count INTEGER NOT NULL DEFAULT 0 CHECK(suggestion_count IN (0,2)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            UNIQUE(account_scope_id,contact_id,local_date,context_hash)
        );
        CREATE INDEX idx_proactive_contact_cache_latest
        ON proactive_contact_cache(account_scope_id,contact_id,local_date,updated_at DESC);
        """,
    ),
    (
        35,
        """
        -- Proactive Suggestion Feedback V1 is local-only writing-style data.
        -- It deliberately references the contact identity but does not write
        -- customer facts, orders, sales state, followups or knowledge tables.
        CREATE TABLE suggestion_feedback (
            feedback_id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            local_date TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            suggestion_kind TEXT NOT NULL DEFAULT 'proactive'
                CHECK(suggestion_kind='proactive'),
            suggestion_index INTEGER NOT NULL CHECK(suggestion_index IN (0,1)),
            reason_type TEXT NOT NULL,
            original_text TEXT NOT NULL,
            final_text TEXT NOT NULL DEFAULT '',
            feedback_type TEXT NOT NULL CHECK(feedback_type IN (
                'accepted','edited_and_accepted','rejected'
            )),
            reason_code TEXT NOT NULL DEFAULT '' CHECK(reason_code IN (
                '', 'too_formal','too_long','too_salesy','wrong_question',
                'missing_history','unnatural_salutation','too_stiff','other'
            )),
            business_context_summary TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            generated_at TEXT NOT NULL,
            accepted_at TEXT,
            regeneration_requested_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            UNIQUE(
                account_scope_id,contact_id,local_date,context_hash,
                suggestion_index,feedback_type,original_text,final_text,reason_code
            )
        );
        CREATE INDEX idx_suggestion_feedback_learning
        ON suggestion_feedback(account_scope_id,reason_type,feedback_type,created_at DESC);
        CREATE INDEX idx_suggestion_feedback_context
        ON suggestion_feedback(account_scope_id,contact_id,local_date,context_hash,updated_at DESC);
        CREATE UNIQUE INDEX idx_suggestion_feedback_regeneration_once
        ON suggestion_feedback(account_scope_id,contact_id,local_date,context_hash)
        WHERE regeneration_requested_at IS NOT NULL;
        """,
    ),
    (
        36,
        """
        -- Old-customer Activation V1 produces exactly one proactive message,
        -- while Reply remains exactly two.  Rebuild only the proactive cache
        -- so its audited count can represent both legitimate modes.
        CREATE TABLE proactive_contact_cache_v36 (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            local_date TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            reason_type TEXT NOT NULL CHECK(reason_type IN (
                'reply_obligation','order_followup','usage_followup','effect_followup',
                'predicted_window','reactivation','do_not_contact_now'
            )),
            decision_status TEXT NOT NULL CHECK(decision_status IN (
                'ineligible','do_not_contact','reply_priority','proactive','blocked'
            )),
            outcome_json TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            suggestion_count INTEGER NOT NULL DEFAULT 0 CHECK(suggestion_count IN (0,1,2)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            UNIQUE(account_scope_id,contact_id,local_date,context_hash)
        );
        INSERT INTO proactive_contact_cache_v36(
            id,account_scope_id,contact_id,local_date,context_hash,reason_type,
            decision_status,outcome_json,provider,model,suggestion_count,created_at,updated_at
        )
        SELECT
            id,account_scope_id,contact_id,local_date,context_hash,reason_type,
            decision_status,outcome_json,provider,model,suggestion_count,created_at,updated_at
        FROM proactive_contact_cache;
        DROP TABLE proactive_contact_cache;
        ALTER TABLE proactive_contact_cache_v36 RENAME TO proactive_contact_cache;
        CREATE INDEX idx_proactive_contact_cache_latest
        ON proactive_contact_cache(account_scope_id,contact_id,local_date,updated_at DESC);
        """,
    ),
    (
        37,
        """
        -- Instant Activation V1 separates a locally usable assessment/message
        -- snapshot from optional AI text.  The fast path reads this table by
        -- exact account/contact identity and never needs to rebuild context or
        -- wait for a provider.
        CREATE TABLE activation_snapshots (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            value_tier TEXT NOT NULL CHECK(value_tier IN ('A','B')),
            status TEXT NOT NULL CHECK(status IN ('ready','blocked','error')),
            stage_json TEXT NOT NULL DEFAULT '{}',
            stage_hash TEXT NOT NULL DEFAULT '',
            need_json TEXT NOT NULL DEFAULT '{}',
            need_hash TEXT NOT NULL DEFAULT '',
            talking_point TEXT NOT NULL DEFAULT '',
            talking_point_hash TEXT NOT NULL DEFAULT '',
            local_text TEXT NOT NULL DEFAULT '',
            local_outcome_json TEXT NOT NULL DEFAULT '',
            local_context_hash TEXT NOT NULL DEFAULT '',
            local_generated_at TEXT NOT NULL,
            local_valid_until TEXT NOT NULL,
            ai_text TEXT NOT NULL DEFAULT '',
            ai_outcome_json TEXT NOT NULL DEFAULT '',
            ai_context_hash TEXT NOT NULL DEFAULT '',
            ai_source TEXT NOT NULL DEFAULT '' CHECK(ai_source IN ('','prewarm','on_demand')),
            ai_provider TEXT NOT NULL DEFAULT '',
            ai_model TEXT NOT NULL DEFAULT '',
            ai_generated_at TEXT,
            ai_valid_until TEXT,
            last_customer_message_at TEXT,
            last_our_message_at TEXT,
            last_contact_at TEXT,
            last_message_fingerprint TEXT NOT NULL DEFAULT '',
            known_reply_pending INTEGER NOT NULL DEFAULT 0 CHECK(known_reply_pending IN (0,1)),
            confidence TEXT NOT NULL DEFAULT 'low' CHECK(confidence IN ('low','medium','high')),
            source_revision TEXT NOT NULL DEFAULT '',
            invalidated_at TEXT,
            invalidation_reason TEXT NOT NULL DEFAULT '',
            last_accessed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
            UNIQUE(account_scope_id,contact_id)
        );
        CREATE INDEX idx_activation_snapshots_fast
        ON activation_snapshots(account_scope_id,contact_id,status,invalidated_at);
        CREATE INDEX idx_activation_snapshots_prewarm
        ON activation_snapshots(value_tier,ai_valid_until,updated_at);

        -- Exact title pixels are learned only after the existing full identity
        -- lock confirms the same contact. A conflicting mapping is quarantined
        -- and never eligible for a zero-OCR fast lookup.
        CREATE TABLE activation_title_fingerprints (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            target_instance_id TEXT NOT NULL,
            exact_title_hash TEXT NOT NULL,
            visual_title_hash TEXT NOT NULL DEFAULT '',
            normalized_title TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
            confirmed_count INTEGER NOT NULL DEFAULT 1 CHECK(confirmed_count > 0),
            conflicted INTEGER NOT NULL DEFAULT 0 CHECK(conflicted IN (0,1)),
            first_confirmed_at TEXT NOT NULL,
            last_confirmed_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            UNIQUE(account_scope_id,target_instance_id,exact_title_hash)
        );
        CREATE INDEX idx_activation_title_fingerprint_lookup
        ON activation_title_fingerprints(account_scope_id,target_instance_id,exact_title_hash,conflicted);
        CREATE INDEX idx_activation_title_fingerprint_contact
        ON activation_title_fingerprints(account_scope_id,contact_id,last_confirmed_at DESC);

        CREATE TABLE activation_prewarm_attempts (
            id TEXT PRIMARY KEY,
            local_date TEXT NOT NULL,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            context_hash TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL CHECK(status IN (
                'started','completed','local_only','paused','skipped','failed'
            )),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            elapsed_ms INTEGER CHECK(elapsed_ms IS NULL OR elapsed_ms >= 0),
            detail TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE
        );
        CREATE INDEX idx_activation_prewarm_attempts_day
        ON activation_prewarm_attempts(local_date,status,started_at);
        CREATE INDEX idx_activation_prewarm_attempts_contact
        ON activation_prewarm_attempts(account_scope_id,contact_id,started_at DESC);

        -- Durable invalidation: changes in messages, memories, profile facts,
        -- events/orders, followups, feedback or the contact itself make the
        -- local assessment stale.  AI expiry is independent and does not make
        -- the local message unavailable.
        CREATE TRIGGER trg_activation_messages_insert AFTER INSERT ON messages BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='message_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_messages_update AFTER UPDATE ON messages BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='message_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_messages_delete AFTER DELETE ON messages BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='message_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=OLD.contact_id;
        END;
        CREATE TRIGGER trg_activation_memories_insert AFTER INSERT ON memories BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='memory_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_memories_update AFTER UPDATE ON memories BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='memory_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_memories_delete AFTER DELETE ON memories BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='memory_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=OLD.contact_id;
        END;
        CREATE TRIGGER trg_activation_contacts_update AFTER UPDATE ON contacts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='contact_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.id;
        END;
        CREATE TRIGGER trg_activation_profiles_update AFTER UPDATE ON customer_profiles BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='profile_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_feedback_insert AFTER INSERT ON suggestion_feedback BEGIN
            UPDATE activation_snapshots SET updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=NEW.contact_id AND account_scope_id=NEW.account_scope_id;
        END;
        CREATE TRIGGER trg_activation_observation_update AFTER UPDATE ON conversation_observations BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='observation_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE conversation_id=NEW.conversation_id;
        END;
        CREATE TRIGGER trg_activation_observation_insert AFTER INSERT ON conversation_observations BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='observation_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE conversation_id=NEW.conversation_id;
        END;
        CREATE TRIGGER trg_activation_observation_delete AFTER DELETE ON conversation_observations BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='observation_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE conversation_id=OLD.conversation_id;
        END;

        CREATE TRIGGER trg_activation_context_insert AFTER INSERT ON customer_growing_contexts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='growing_context_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_context_update AFTER UPDATE ON customer_growing_contexts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='growing_context_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_context_delete AFTER DELETE ON customer_growing_contexts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='growing_context_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=OLD.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_fact_insert AFTER INSERT ON customer_facts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='fact_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_fact_update AFTER UPDATE ON customer_facts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='fact_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_fact_delete AFTER DELETE ON customer_facts BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='fact_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=OLD.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_event_insert AFTER INSERT ON customer_events BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='event_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_event_update AFTER UPDATE ON customer_events BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='event_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_event_delete AFTER DELETE ON customer_events BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='event_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=OLD.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_followup_insert AFTER INSERT ON customer_followups BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='followup_insert',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_followup_update AFTER UPDATE ON customer_followups BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='followup_update',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=NEW.customer_profile_id);
        END;
        CREATE TRIGGER trg_activation_followup_delete AFTER DELETE ON customer_followups BEGIN
            UPDATE activation_snapshots SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='followup_delete',updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE contact_id=(SELECT contact_id FROM customer_profiles WHERE id=OLD.customer_profile_id);
        END;
        """,
    ),
    (
        38,
        """
        -- Customer Conversation Style V1 is customer/account scoped expression
        -- metadata.  It is derived only from our outgoing messages (plus a
        -- deliberately weak edit-feedback signal) and is not a Customer Fact.
        CREATE TABLE customer_conversation_styles (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            preferred_salutation TEXT NOT NULL DEFAULT '',
            salutation_confidence TEXT NOT NULL DEFAULT 'low'
                CHECK(salutation_confidence IN ('low','medium','high')),
            salutation_evidence_count INTEGER NOT NULL DEFAULT 0
                CHECK(salutation_evidence_count >= 0),
            salutation_recent_count INTEGER NOT NULL DEFAULT 0
                CHECK(salutation_recent_count >= 0),
            salutation_current_year_count INTEGER NOT NULL DEFAULT 0
                CHECK(salutation_current_year_count >= 0),
            outgoing_sample_count INTEGER NOT NULL DEFAULT 0
                CHECK(outgoing_sample_count >= 0),
            last_salutation_used_at TEXT,
            prefers_no_salutation INTEGER NOT NULL DEFAULT 0
                CHECK(prefers_no_salutation IN (0,1)),
            manual_preferred_salutation TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '{}',
            evidence_hash TEXT NOT NULL DEFAULT '',
            style_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            UNIQUE(account_scope_id,contact_id)
        );
        CREATE INDEX idx_customer_conversation_styles_contact
        ON customer_conversation_styles(account_scope_id,contact_id,updated_at DESC);

        -- Keep the click path self-contained: these fields are copied into the
        -- lightweight Activation Snapshot during provider-free background work.
        ALTER TABLE activation_snapshots
            ADD COLUMN preferred_salutation TEXT NOT NULL DEFAULT '';
        ALTER TABLE activation_snapshots
            ADD COLUMN salutation_confidence TEXT NOT NULL DEFAULT 'low'
                CHECK(salutation_confidence IN ('low','medium','high'));
        ALTER TABLE activation_snapshots
            ADD COLUMN conversation_style_version TEXT NOT NULL DEFAULT '';
        ALTER TABLE activation_snapshots
            ADD COLUMN salutation_evidence_hash TEXT NOT NULL DEFAULT '';

        -- A perceptual title hash is only eligible when it resolves uniquely
        -- inside the same account/window instance.  The application still
        -- keeps the existing two-frame lock before publishing identity.
        CREATE INDEX idx_activation_title_fingerprint_visual
        ON activation_title_fingerprints(
            account_scope_id,target_instance_id,visual_title_hash,conflicted
        );

        CREATE TRIGGER trg_activation_style_insert
        AFTER INSERT ON customer_conversation_styles BEGIN
            UPDATE activation_snapshots
            SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='conversation_style_insert',
                updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE account_scope_id=NEW.account_scope_id AND contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_style_update
        AFTER UPDATE ON customer_conversation_styles BEGIN
            UPDATE activation_snapshots
            SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='conversation_style_update',
                updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE account_scope_id=NEW.account_scope_id AND contact_id=NEW.contact_id;
        END;
        CREATE TRIGGER trg_activation_style_delete
        AFTER DELETE ON customer_conversation_styles BEGIN
            UPDATE activation_snapshots
            SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
                invalidation_reason='conversation_style_delete',
                updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now')
            WHERE account_scope_id=OLD.account_scope_id AND contact_id=OLD.contact_id;
        END;

        -- Every v37 snapshot lacks customer-specific salutation metadata.
        UPDATE activation_snapshots
        SET invalidated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'),
            invalidation_reason='conversation_style_v1_upgrade',
            updated_at=STRFTIME('%Y-%m-%dT%H:%M:%fZ','now');
        """,
    ),
    (
        39,
        """
        -- Instant Activation is Contact-scoped.  Conversation remains
        -- optional and belongs to the message/Reply path; deleting one must
        -- not delete the Contact's provider-free activation snapshot.
        PRAGMA legacy_alter_table=ON;
        DROP INDEX idx_activation_snapshots_fast;
        DROP INDEX idx_activation_snapshots_prewarm;
        ALTER TABLE activation_snapshots RENAME TO activation_snapshots_v38;
        CREATE TABLE activation_snapshots (
            id TEXT PRIMARY KEY,
            account_scope_id TEXT NOT NULL,
            contact_id TEXT NOT NULL,
            conversation_id TEXT,
            value_tier TEXT NOT NULL CHECK(value_tier IN ('A','B')),
            status TEXT NOT NULL CHECK(status IN ('ready','blocked','error')),
            stage_json TEXT NOT NULL DEFAULT '{}',
            stage_hash TEXT NOT NULL DEFAULT '',
            need_json TEXT NOT NULL DEFAULT '{}',
            need_hash TEXT NOT NULL DEFAULT '',
            talking_point TEXT NOT NULL DEFAULT '',
            talking_point_hash TEXT NOT NULL DEFAULT '',
            local_text TEXT NOT NULL DEFAULT '',
            local_outcome_json TEXT NOT NULL DEFAULT '',
            local_context_hash TEXT NOT NULL DEFAULT '',
            local_generated_at TEXT NOT NULL,
            local_valid_until TEXT NOT NULL,
            ai_text TEXT NOT NULL DEFAULT '',
            ai_outcome_json TEXT NOT NULL DEFAULT '',
            ai_context_hash TEXT NOT NULL DEFAULT '',
            ai_source TEXT NOT NULL DEFAULT '' CHECK(ai_source IN ('','prewarm','on_demand')),
            ai_provider TEXT NOT NULL DEFAULT '',
            ai_model TEXT NOT NULL DEFAULT '',
            ai_generated_at TEXT,
            ai_valid_until TEXT,
            last_customer_message_at TEXT,
            last_our_message_at TEXT,
            last_contact_at TEXT,
            last_message_fingerprint TEXT NOT NULL DEFAULT '',
            known_reply_pending INTEGER NOT NULL DEFAULT 0 CHECK(known_reply_pending IN (0,1)),
            confidence TEXT NOT NULL DEFAULT 'low' CHECK(confidence IN ('low','medium','high')),
            source_revision TEXT NOT NULL DEFAULT '',
            invalidated_at TEXT,
            invalidation_reason TEXT NOT NULL DEFAULT '',
            last_accessed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            preferred_salutation TEXT NOT NULL DEFAULT '',
            salutation_confidence TEXT NOT NULL DEFAULT 'low'
                CHECK(salutation_confidence IN ('low','medium','high')),
            conversation_style_version TEXT NOT NULL DEFAULT '',
            salutation_evidence_hash TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(account_scope_id) REFERENCES account_scopes(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
            UNIQUE(account_scope_id,contact_id)
        );
        INSERT INTO activation_snapshots(
            id,account_scope_id,contact_id,conversation_id,value_tier,status,
            stage_json,stage_hash,need_json,need_hash,talking_point,talking_point_hash,
            local_text,local_outcome_json,local_context_hash,local_generated_at,local_valid_until,
            ai_text,ai_outcome_json,ai_context_hash,ai_source,ai_provider,ai_model,
            ai_generated_at,ai_valid_until,last_customer_message_at,last_our_message_at,
            last_contact_at,last_message_fingerprint,known_reply_pending,confidence,
            source_revision,invalidated_at,invalidation_reason,last_accessed_at,created_at,updated_at,
            preferred_salutation,salutation_confidence,conversation_style_version,salutation_evidence_hash
        )
        SELECT
            id,account_scope_id,contact_id,NULLIF(conversation_id,''),value_tier,status,
            stage_json,stage_hash,need_json,need_hash,talking_point,talking_point_hash,
            local_text,local_outcome_json,local_context_hash,local_generated_at,local_valid_until,
            ai_text,ai_outcome_json,ai_context_hash,ai_source,ai_provider,ai_model,
            ai_generated_at,ai_valid_until,last_customer_message_at,last_our_message_at,
            last_contact_at,last_message_fingerprint,known_reply_pending,confidence,
            source_revision,invalidated_at,invalidation_reason,last_accessed_at,created_at,updated_at,
            preferred_salutation,salutation_confidence,conversation_style_version,salutation_evidence_hash
        FROM activation_snapshots_v38;
        DROP TABLE activation_snapshots_v38;
        CREATE INDEX idx_activation_snapshots_fast
        ON activation_snapshots(account_scope_id,contact_id,status,invalidated_at);
        CREATE INDEX idx_activation_snapshots_prewarm
        ON activation_snapshots(value_tier,ai_valid_until,updated_at);
        PRAGMA legacy_alter_table=OFF;
        """,
    ),
]
