"""Tests for marine_kb.py — MarineKnowledgeBase."""

import pytest
from jetson.knowledge_graph.marine_kb import MarineKnowledgeBase


# ======================================================================
# Bootstrap / Initialization
# ======================================================================

class TestMarineKBInit:
    def test_creates_vessel_types(self):
        kb = MarineKnowledgeBase()
        vessels = kb.find_entities(type_filter="VesselType")
        assert len(vessels) >= 8

    def test_creates_equipment(self):
        kb = MarineKnowledgeBase()
        equips = kb.find_entities(type_filter="Equipment")
        assert len(equips) >= 6

    def test_creates_regulations(self):
        kb = MarineKnowledgeBase()
        regs = kb.find_entities(type_filter="Regulation")
        assert len(regs) >= 5

    def test_creates_situations(self):
        kb = MarineKnowledgeBase()
        situations = kb.find_entities(type_filter="Situation")
        assert len(situations) >= 4

    def test_has_relations(self):
        kb = MarineKnowledgeBase()
        assert kb.stats()["relation_count"] > 0

    def test_stats_entity_types(self):
        kb = MarineKnowledgeBase()
        types = kb.stats()["entity_types"]
        assert "VesselType" in types
        assert "Equipment" in types
        assert "Regulation" in types


# ======================================================================
# Vessel types
# ======================================================================

class TestVesselTypes:
    def test_asv_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("vessel:autonomous_surface_vehicle")
        assert entity is not None
        assert entity.type == "VesselType"

    def test_rov_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("vessel:remotely_operated_vehicle")
        assert entity is not None

    def test_auv_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("vessel:autonomous_underwater_vehicle")
        assert entity is not None

    def test_asv_has_autonomy_full(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("vessel:autonomous_surface_vehicle")
        assert entity.properties["autonomy_level"] == "full"

    def test_vessel_has_labels(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("vessel:cargo_ship")
        assert "marine" in entity.labels
        assert "vessel" in entity.labels

    def test_asv_uses_equipment(self):
        kb = MarineKnowledgeBase()
        rels = kb.get_relations("vessel:autonomous_surface_vehicle", "outgoing")
        uses = [r for r in rels if r.type == "uses"]
        assert len(uses) >= 3


# ======================================================================
# Equipment
# ======================================================================

class TestEquipment:
    def test_sonar_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("equip:sonar_array")
        assert entity is not None
        assert entity.type == "Equipment"

    def test_gps_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("equip:gps_receiver")
        assert entity is not None

    def test_equipment_has_labels(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("equip:sonar_array")
        assert "sensor" in entity.labels


# ======================================================================
# Regulations
# ======================================================================

class TestRegulations:
    def test_colreg_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("reg:colreg_rule_5")
        assert entity is not None

    def test_solhas_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("reg:solas_chapter_v")
        assert entity is not None

    def test_regulation_applies_to_vessels(self):
        kb = MarineKnowledgeBase()
        rels = kb.get_relations("reg:colreg_rule_5", "outgoing")
        applies = [r for r in rels if r.type == "applies_to"]
        assert len(applies) >= 3

    def test_imo_code_for_autonomous(self):
        kb = MarineKnowledgeBase()
        rels = kb.get_relations("reg:imo_autonomous_ships_code", "outgoing")
        applies = [r for r in rels if r.type == "applies_to"]
        targets = {r.target_id for r in applies}
        assert "vessel:autonomous_surface_vehicle" in targets


# ======================================================================
# Situations
# ======================================================================

class TestSituations:
    def test_collision_risk_exists(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("situation:collision_risk")
        assert entity is not None
        assert entity.properties["severity"] == "high"

    def test_equipment_failure_severity(self):
        kb = MarineKnowledgeBase()
        entity = kb.get_entity("situation:equipment_failure")
        assert entity.properties["severity"] == "critical"


# ======================================================================
# Convenience methods
# ======================================================================

class TestConvenienceMethods:
    def test_add_vessel_type(self):
        kb = MarineKnowledgeBase()
        e = kb.add_vessel_type("Hovercraft", {"speed_knots": 40})
        assert e.id == "vessel:hovercraft"
        assert e.type == "VesselType"

    def test_add_equipment(self):
        kb = MarineKnowledgeBase()
        e = kb.add_equipment("Thermal Camera", {"resolution": "640x480"})
        assert e.id == "equip:thermal_camera"
        assert e.type == "Equipment"

    def test_add_regulation(self):
        kb = MarineKnowledgeBase()
        e = kb.add_regulation("Rule 10", {"topic": "navigation"})
        assert e.type == "Regulation"

    def test_add_marine_relation(self):
        kb = MarineKnowledgeBase()
        kb.add_vessel_type("Submarine")
        kb.add_equipment("Periscope")
        r = kb.add_marine_relation("vessel:submarine", "equip:periscope", "uses")
        assert r.type == "uses"

    def test_get_equipment_for_vessel(self):
        kb = MarineKnowledgeBase()
        equips = kb.get_equipment_for_vessel("Autonomous Surface Vehicle")
        assert len(equips) >= 3
        names = {e.id for e in equips}
        assert "equip:gps_receiver" in names

    def test_get_equipment_for_vessel_empty(self):
        kb = MarineKnowledgeBase()
        kb.add_vessel_type("Unknown Vessel")
        equips = kb.get_equipment_for_vessel("Unknown Vessel")
        assert equips == []

    def test_get_regulations_for_situation(self):
        kb = MarineKnowledgeBase()
        regs = kb.get_regulations_for_situation("collision_risk")
        assert len(regs) >= 1
        types = {r.type for r in regs}
        assert "Regulation" in types

    def test_get_regulations_for_situation_empty(self):
        kb = MarineKnowledgeBase()
        regs = kb.get_regulations_for_situation("nonexistent_sit")
        assert regs == []


# ======================================================================
# Query maritime knowledge
# ======================================================================

class TestQueryMaritimeKnowledge:
    def test_query_vessel_types(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("vessel_types")
        assert len(results) >= 8

    def test_query_equipment(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("equipment")
        assert len(results) >= 6

    def test_query_regulations(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("regulations")
        assert len(results) >= 5

    def test_query_situations(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("situations")
        assert len(results) >= 4

    def test_query_vessel_by_autonomy(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("vessel_by_autonomy", level="full")
        assert len(results) >= 2
        for e in results:
            assert e.properties["autonomy_level"] == "full"

    def test_query_equipment_by_type(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("equipment_by_type", etype="acoustic")
        assert len(results) >= 2

    def test_query_unknown_type(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("nonexistent")
        assert results == []

    def test_query_vessel_by_autonomy_no_match(self):
        kb = MarineKnowledgeBase()
        results = kb.query_maritime_knowledge("vessel_by_autonomy", level="nonexistent")
        assert results == []
