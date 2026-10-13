"""
Modèles de tables relationnelles SQLAlchemy 2.0 pour PostgreSQL.
Correspondance exacte avec le schéma DDL du projet.
"""
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Numeric,
    Boolean,
    Text,
    Date,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Offre(Base):
    """
    Table des Offres d'Emploi.
    Source de vérité pour toutes les données scrapées et normalisées.
    """
    __tablename__ = "offres"

    id = Column(String(64), primary_key=True, index=True, comment="Hash unique job_hash")
    poste = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(255), default="France", index=True)
    type_de_contrat = Column(String(50), nullable=False, index=True)  # CDI, CDD, Stage...
    niveau_etude_min = Column(Integer, default=0, index=True)        # 0 (indifférent), 3 (Bac+3), 5 (Bac+5)
    min_experience_years = Column(Numeric(4, 1), default=0.0)
    
    # Compétences requises stockées sous forme de tableau PostgreSQL requêtable avec @>
    competences_necessaires = Column(ARRAY(Text), nullable=False, default=list)
    
    description = Column(Text, nullable=False)
    mission = Column(Text, nullable=False)
    source_url = Column(Text, default="")
    source = Column(String(50), default="unknown", index=True, comment="Origine: adzuna, indeed, linkedin")
    secteur = Column(String(50), default="it_data", index=True, comment="Secteur: it_data, sante_medical, etc.")
    date_posted = Column(Date, default=func.current_date(), index=True, comment="Date de publication")
    is_active = Column(Boolean, default=True, index=True)

    # Suivi d'indexation vectorielle (Architecture ELT Découplée)
    is_indexed = Column(Boolean, default=False, index=True, comment="True si vectorisé dans ChromaDB")
    indexed_at = Column(DateTime(timezone=True), nullable=True, comment="Date de l'indexation vectorielle")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    evaluations = relationship("Evaluation", back_populates="offre", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Offre(id='{self.id}', poste='{self.poste}', company='{self.company}', contrat='{self.type_de_contrat}')>"


class CVUser(Base):
    """
    Table des Candidats (Profils de CVs extraits par Docling et Pydantic).
    """
    __tablename__ = "cv_users"

    id = Column(String(64), primary_key=True, index=True, comment="Identifiant unique cand_xxx")
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    
    desired_contract_types = Column(ARRAY(String(50)), nullable=False, default=list)
    parcours_academique = Column(Integer, nullable=False, default=5)  # 3 (Bac+3), 5 (Bac+5)
    total_experience_years = Column(Numeric(4, 1), default=0.0)
    
    competences = Column(ARRAY(Text), nullable=False, default=list)
    experiences_professionnelles = Column(JSONB, default=list)
    experiences_universitaires = Column(JSONB, default=list)
    
    raw_markdown = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relations
    evaluations = relationship("Evaluation", back_populates="cv_user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<CVUser(id='{self.id}', name='{self.full_name}', bac='{self.parcours_academique}')>"


class Evaluation(Base):
    """
    Table d'association Many-to-Many entre Candidats et Offres.
    Conserve les scores décomposés (Dense, BM25, Reranker) et l'explicabilité LLM.
    """
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cv_user_id = Column(String(64), ForeignKey("cv_users.id", ondelete="CASCADE"), nullable=False, index=True)
    offre_id = Column(String(64), ForeignKey("offres.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Décomposition des scores pour l'évaluation et l'auditabilité
    dense_score = Column(Float, nullable=True)
    bm25_score = Column(Float, nullable=True)
    reranker_score = Column(Float, nullable=True)
    final_score = Column(Integer, nullable=False, index=True)  # Score consolidé (0 à 100)
    
    # Explicabilité
    matched_skills = Column(ARRAY(Text), default=list)
    missing_skills = Column(ARRAY(Text), default=list)
    justification = Column(Text, nullable=True)
    conseils_candidature = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Contrainte d'unicité : 1 seule évaluation par couple (CV, Offre)
    __table_args__ = (
        UniqueConstraint("cv_user_id", "offre_id", name="uq_eval_cv_offre"),
    )

    # Relations inverses
    cv_user = relationship("CVUser", back_populates="evaluations")
    offre = relationship("Offre", back_populates="evaluations")

    def __repr__(self) -> str:
        return f"<Evaluation(cv='{self.cv_user_id}', offre='{self.offre_id}', score={self.final_score})>"
