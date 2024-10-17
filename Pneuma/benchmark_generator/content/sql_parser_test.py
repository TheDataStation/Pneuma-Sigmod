import sql_parser

def main():
    sql = 'select product, count(distinct quantity) from product_sale where sale_date between "2024-01-01" and "2024-06-30" and catgeory = "Beauty" and store_location = "New York"  group by product having sum(quanlity) > 10 order by quanlity desc limit 3 '

    sql_2 = ' \nSELECT "borough", "cohort", "local_#", "cohort_#"\nFROM table_name\nWHERE "tasc_(ged)_#" = 20 AND "sacc_(iep_diploma)_#" = 60\nORDER BY "advanced_regents_#" DESC\nLIMIT 5;\n'
    
    sql_3 = 'SELECT SUM("cohort_#")  FROM table_name  WHERE cohort_year = 2013'

    sql_4 = 'select max("total_regents_%_of_grads") as "max_total_regents_percentage", max("tasc_(ged)_%_of_cohort") as "max_ged_percentage" from "2001- 2013 Graduation Outcomes Borough- ALL STUDENTS,SWD,GENDER,ELL,ETHNICITY,EVER ELL" where "dropout_#" between 2618 and 3724'

    stmt = sql_parser.parse_sql(sql_4)
    import pdb; pdb.set_trace()
    # stmt.args: 'expressions', 'where', 'group', 'having', 'order', 'limit',
    print("select  ", sql_parser.get_select(stmt))
    print("where  ", sql_parser.get_where(stmt))
    print("group by ", sql_parser.get_group_by(stmt))
    print("having ", sql_parser.get_having(stmt))
    print("order by", sql_parser.get_order_by(stmt))
    print("limit", sql_parser.get_limit(stmt))

if __name__ == '__main__':
    main()