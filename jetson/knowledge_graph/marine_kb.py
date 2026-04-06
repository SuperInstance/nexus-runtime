"""Marine domain knowledge base for NEXUS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph import Entity, KnowledgeGraph, Relation, make_entity, make_relation


class MarineKnowledgeBase(KnowledgeGraph):
    """Pre-populated knowledge graph for marine / maritime domain."""

    def __init__(self) -> None:
        super().__init__()
        self._bootstrap()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Pre-populate with ~20 marine entities and relations."""

        # --- Vessel types ---
        vessel_ids: Dict[str, str] = {}
        vessel_data = [
            ("Autonomous Surface Vehicle", {"length_m": 12.0, "displacement_t": 5.0, "autonomy_level": "full"}),
            ("Remotely Operated Vehicle", {"length_m": 3.0, "depth_rating_m": 4000, "autonomy_level": "remote"}),
            ("Autonomous Underwater Vehicle", {"length_m": 5.0, "depth_rating_m": 6000, "autonomy_level": "full"}),
            ("Cargo Ship", {"length_m": 300.0, "displacement_t": 50000, "autonomy_level": "manual"}),
            ("Tugboat", {"length_m": 30.0, "displacement_t": 500, "autonomy_level": "manual"}),
            ("Research Vessel", {"length_m": 80.0, "displacement_t": 4000, "autonomy_level": "semi"}),
            ("Patrol Boat", {"length_m": 20.0, "displacement_t": 50, "autonomy_level": "semi"}),
            ("Barge", {"length_m": 60.0, "displacement_t": 2000, "autonomy_level": "manual"}),
        ]
        for name, props in vessel_data:
            eid = f"vessel:{name.lower().replace(' ', '_')}"
            vessel_ids[name] = eid
            self.add_entity(make_entity("VesselType", props, labels=("marine", "vessel"), entity_id=eid))

        # --- Equipment types ---
        equip_ids: Dict[str, str] = {}
        equip_data = [
            ("Sonar Array", {"frequency_khz": 12, "range_m": 500, "type": "active"}),
            ("LiDAR Sensor", {"range_m": 200, "accuracy_m": 0.02, "type": "laser"}),
            ("GPS Receiver", {"accuracy_m": 1.5, "constellation": "multi-gnss", "type": "gnss"}),
            ("IMU", {"accuracy_deg_hr": 0.01, "axes": 6, "type": "inertial"}),
            ("Doppler Velocity Log", {"accuracy_m_s": 0.001, "type": "acoustic"}),
            ("Multibeam Echosounder", {"frequency_khz": 100, "swath_deg": 120, "type": "acoustic"}),
        ]
        for name, props in equip_data:
            eid = f"equip:{name.lower().replace(' ', '_')}"
            equip_ids[name] = eid
            self.add_entity(make_entity("Equipment", props, labels=("sensor", "marine"), entity_id=eid))

        # --- Regulations ---
        reg_ids: Dict[str, str] = {}
        reg_data = [
            ("COLREG Rule 5", {"jurisdiction": "international", "topic": "lookout", "chapter": "II"}),
            ("COLREG Rule 7", {"jurisdiction": "international", "topic": "risk_assessment", "chapter": "II"}),
            ("SOLAS Chapter V", {"jurisdiction": "international", "topic": "safety_navigation", "chapter": "V"}),
            ("IMO Autonomous Ships Code", {"jurisdiction": "international", "topic": "autonomous_vessels", "chapter": "MSC"}),
            ("MARPOL Annex VI", {"jurisdiction": "international", "topic": "emissions", "chapter": "VI"}),
        ]
        for name, props in reg_data:
            eid = f"reg:{name.lower().replace(' ', '_').replace('.', '')}"
            reg_ids[name] = eid
            self.add_entity(make_entity("Regulation", props, labels=("maritime_law",), entity_id=eid))

        # --- Relations: vessels use equipment ---
        vessel_equip_map = {
            "Autonomous Surface Vehicle": ["Sonar Array", "GPS Receiver", "IMU", "LiDAR Sensor"],
            "Remotely Operated Vehicle": ["Sonar Array", "IMU", "Doppler Velocity Log"],
            "Autonomous Underwater Vehicle": ["Sonar Array", "IMU", "Doppler Velocity Log", "Multibeam Echosounder"],
            "Cargo Ship": ["GPS Receiver", "IMU"],
            "Tugboat": ["GPS Receiver", "IMU", "Sonar Array"],
            "Research Vessel": ["Sonar Array", "GPS Receiver", "IMU", "LiDAR Sensor", "Multibeam Echosounder", "Doppler Velocity Log"],
            "Patrol Boat": ["GPS Receiver", "Sonar Array", "LiDAR Sensor"],
            "Barge": ["GPS Receiver"],
        }
        for vessel, equips in vessel_equip_map.items():
            vid = vessel_ids[vessel]
            for eq in equips:
                eid = equip_ids[eq]
                self.add_relation(make_relation(vid, eid, "uses", weight=0.9))

        # --- Relations: regulations apply to vessel types ---
        reg_vessel_map = {
            "COLREG Rule 5": ["Autonomous Surface Vehicle", "Cargo Ship", "Tugboat", "Research Vessel", "Patrol Boat"],
            "COLREG Rule 7": ["Autonomous Surface Vehicle", "Autonomous Underwater Vehicle", "Remotely Operated Vehicle", "Cargo Ship"],
            "SOLAS Chapter V": ["Cargo Ship", "Research Vessel", "Tugboat"],
            "IMO Autonomous Ships Code": ["Autonomous Surface Vehicle", "Autonomous Underwater Vehicle", "Remotely Operated Vehicle"],
            "MARPOL Annex VI": ["Cargo Ship", "Tugboat", "Research Vessel", "Barge"],
        }
        for reg, vessels in reg_vessel_map.items():
            rid = reg_ids[reg]
            for vessel in vessels:
                vid = vessel_ids[vessel]
                self.add_relation(make_relation(rid, vid, "applies_to", weight=0.8))

        # --- Taxonomy: hierarchical ---
        self.add_relation(make_relation(vessel_ids["Autonomous Surface Vehicle"], "vessel:unmanned_vehicle", "is_a"))
        self.add_relation(make_relation(vessel_ids["Autonomous Underwater Vehicle"], "vessel:unmanned_vehicle", "is_a"))
        self.add_relation(make_relation(vessel_ids["Remotely Operated Vehicle"], "vessel:unmanned_vehicle", "is_a"))

        # --- Situations ---
        situations = [
            ("collision_risk", {"severity": "high", "response": "evasive_action"}),
            ("low_visibility", {"severity": "medium", "response": "reduce_speed"}),
            ("equipment_failure", {"severity": "critical", "response": "safe_mode"}),
            ("restricted_waters", {"severity": "medium", "response": "increased_caution"}),
        ]
        sit_ids = {}
        for name, props in situations:
            eid = f"situation:{name}"
            sit_ids[name] = eid
            self.add_entity(make_entity("Situation", props, labels=("operational",), entity_id=eid))

        # Regulations relevant to situations
        self.add_relation(make_relation(reg_ids["COLREG Rule 5"], sit_ids["collision_risk"], "relevant_to"))
        self.add_relation(make_relation(reg_ids["COLREG Rule 7"], sit_ids["collision_risk"], "relevant_to"))
        self.add_relation(make_relation(reg_ids["COLREG Rule 5"], sit_ids["low_visibility"], "relevant_to"))
        self.add_relation(make_relation(reg_ids["SOLAS Chapter V"], sit_ids["equipment_failure"], "relevant_to"))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def add_vessel_type(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        eid = f"vessel:{name.lower().replace(' ', '_')}"
        entity = make_entity("VesselType", properties or {}, labels=("marine", "vessel"), entity_id=eid)
        self.add_entity(entity)
        return entity

    def add_equipment(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        eid = f"equip:{name.lower().replace(' ', '_')}"
        entity = make_entity("Equipment", properties or {}, labels=("sensor", "marine"), entity_id=eid)
        self.add_entity(entity)
        return entity

    def add_regulation(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        eid = f"reg:{name.lower().replace(' ', '_').replace('.', '')}"
        entity = make_entity("Regulation", properties or {}, labels=("maritime_law",), entity_id=eid)
        self.add_entity(entity)
        return entity

    def add_marine_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        weight: float = 1.0,
    ) -> Relation:
        return self.add_relation(make_relation(source_id, target_id, rel_type, weight=weight))

    def get_equipment_for_vessel(self, vessel_name: str) -> List[Entity]:
        eid = f"vessel:{vessel_name.lower().replace(' ', '_')}"
        relations = self.get_relations(eid, "outgoing")
        result: List[Entity] = []
        for rel in relations:
            if rel.type == "uses":
                entity = self.get_entity(rel.target_id)
                if entity is not None:
                    result.append(entity)
        return result

    def get_regulations_for_situation(self, situation_name: str) -> List[Entity]:
        eid = f"situation:{situation_name}"
        relations = self.get_relations(eid, "incoming")
        result: List[Entity] = []
        for rel in relations:
            if rel.type == "relevant_to":
                entity = self.get_entity(rel.source_id)
                if entity is not None:
                    result.append(entity)
        return result

    def query_maritime_knowledge(
        self,
        query_type: str,
        **kwargs: Any,
    ) -> List[Entity]:
        """General-purpose query interface.

        query_type: 'vessel_types', 'equipment', 'regulations', 'situations',
                     'vessel_by_autonomy', 'equipment_by_type'
        """
        if query_type == "vessel_types":
            return self.find_entities(type_filter="VesselType")
        elif query_type == "equipment":
            return self.find_entities(type_filter="Equipment")
        elif query_type == "regulations":
            return self.find_entities(type_filter="Regulation")
        elif query_type == "situations":
            return self.find_entities(type_filter="Situation")
        elif query_type == "vessel_by_autonomy":
            return self.find_entities(type_filter="VesselType", properties={"autonomy_level": kwargs.get("level", "")})
        elif query_type == "equipment_by_type":
            return self.find_entities(type_filter="Equipment", properties={"type": kwargs.get("etype", "")})
        return []
